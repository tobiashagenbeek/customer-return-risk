#!/usr/bin/env php
<?php
/**
 * @author Tobias Hagenbeek <tobias@tobivision.co|tobias@solphit.com>
 */
declare(strict_types=1);

/**
 * ProgressBarCLI
 * - Works in PHP CLI only.
 * - Provides ETA, elapsed, rate, memory, ANSI colors, spinner, auto-fit width.
 * - Call update($done, $total = null, $status = null).
 * - Call finish() when done to commit a trailing newline.
 */
final class ProgressBarCLI
{
    /** @var array<string, mixed> */
    private array $opt;

    private int $total = 0;
    private int $done = 0;
    private float $startedAt;
    private float $lastRedraw = 0.0;
    private bool $finished = false;
    private bool $isTty = true;
    private int $termWidth = 80;
    private int $spinnerIndex = 0;

    /** For smoothed ETA (sliding window) */
    private float $windowSeconds = 5.0;
    /** @var array<int, array{t: float, d: int}> */
    private array $history = [];

    /**
     * @param array<string, mixed> $options
     * Options (all optional):
     * - width: int (max bar width hint; auto-fit trims it). Default 40.
     * - min_interval: float seconds between redraws. Default 0.05
     * - show_percent: bool, default true
     * - show_counters: bool, default true
     * - show_elapsed: bool, default true
     * - show_eta: bool, default true
     * - show_rate: bool, default true  (items/sec)
     * - show_memory: bool, default true
     * - show_peak_memory: bool, default false
     * - show_spinner: bool, default true
     * - colorize: bool, default auto (true if TTY)
     * - fg_done: string ANSI color code (e.g., '32' green), default '32'
     * - fg_empty: string ANSI code for empty bar, default '90'
     * - fg_text: string ANSI code for text, default '37'
     * - bar_char: string (char used for filled), default '='
     * - head_char: string (char used at the tip), default '>'
     * - empty_char: string (empty fill), default ' '
     * - left_delim: string, default '['
     * - right_delim: string, default ']'
     * - spinner: string[] characters for spinner, default ['|','/','-','\\']
     * - label: string initial label/status prefix
     */
    public function __construct(int $total = 0, array $options = [])
    {
        $this->startedAt = microtime(true);
        $this->detectTtyAndWidth();
        $this->opt = $this->applyDefaults($options);
        if (isset($options['colorize'])) {
            $this->opt['colorize'] = (bool)$options['colorize'];
        } else {
            $this->opt['colorize'] = $this->isTty;
        }
        $this->total = max(0, $total);
        $this->installSignalHandlerIfAvailable();
    }

    public function setTotal(int $total): void
    {
        $this->total = max(0, $total);
    }

    public function setLabel(?string $label): void
    {
        $this->opt['label'] = $label ?? '';
    }

    /**
     * Update the bar.
     * @param int $done   How far we are.
     * @param int|null $total Optional new total.
     * @param string|null $status Optional status text to show.
     */
    public function update(int $done, ?int $total = null, ?string $status = null): void
    {
        if ($this->finished) {
            return;
        }
        if ($total !== null) {
            $this->setTotal($total);
        }
        $this->done = max(0, $done);

        if ($status !== null) {
            $this->opt['status'] = $status;
        }

        $now = microtime(true);
        $this->history[] = ['t' => $now, 'd' => $this->done];
        $this->pruneHistory($now);

        // Throttle redraws
        if (($now - $this->lastRedraw) < (float)$this->opt['min_interval'] && $this->done < $this->total) {
            return;
        }
        $this->lastRedraw = $now;

        $this->render();
    }

    /**
     * Increase progress by $inc. Optionally set a status.
     */
    public function tick(int $inc = 1, ?string $status = null): void
    {
        $this->update($this->done + $inc, null, $status);
    }

    /**
     * Mark complete and render final line with newline.
     */
    public function finish(?string $status = null): void
    {
        if ($this->finished) {
            return;
        }
        if ($status !== null) {
            $this->opt['status'] = $status;
        }
        // Force final redraw at 100% if total > 0, else draw one last state.
        if ($this->total > 0) {
            $this->done = $this->total;
        }
        $this->render(true);
        $this->finished = true;
        $this->write(PHP_EOL);
    }

    /* ========================== Internals ========================== */

    /** @return array<string, mixed> */
    private function applyDefaults(array $o): array
    {
        $d = [
            'width' => 40,
            'min_interval' => 0.05,
            'show_percent' => true,
            'show_counters' => true,
            'show_elapsed' => true,
            'show_eta' => true,
            'show_rate' => true,
            'show_memory' => true,
            'show_peak_memory' => false,
            'show_spinner' => true,
            'colorize' => true,
            'fg_done' => '32',   // green
            'fg_empty' => '90',  // bright black/gray
            'fg_text' => '37',   // white
            'bar_char' => '=',
            'head_char' => '>',
            'empty_char' => ' ',
            'left_delim' => '[',
            'right_delim' => ']',
            'spinner' => ['|', '/', '-', '\\'],
            'label' => '',
            'status' => '',
        ];
        return $o + $d;
    }

    private function detectTtyAndWidth(): void
    {
        $this->isTty = $this->detectTty(STDOUT);
        $this->termWidth = $this->detectWidth();
    }

    private function detectTty($stream): bool
    {
        // Prefer posix_isatty if available
        if (function_exists('posix_isatty')) {
            /** @phpstan-ignore-next-line */
            return @posix_isatty($stream) === true;
        }
        // Fallback heuristic: if STDOUT is defined and is a resource, assume TTY
        // (Won’t be perfect for redirected output; user can override colorize=false)
        return true;
    }

    private function detectWidth(): int
    {
        // Priority: env COLUMNS -> tput -> stty -> default 80
        $env = getenv('COLUMNS');
        if ($env !== false && ctype_digit($env)) {
            $w = (int)$env;
            if ($w > 0) return $w;
        }
        $w = $this->execInt('tput cols');
        if ($w > 0) return $w;

        $stty = $this->execString('stty size');
        if ($stty) {
            $parts = preg_split('/\s+/', trim($stty));
            if ($parts && count($parts) === 2 && ctype_digit($parts[1])) {
                return max(20, (int)$parts[1]);
            }
        }
        return 80;
    }

    private function execInt(string $cmd): int
    {
        $out = $this->execString($cmd);
        return ($out && ctype_digit($out)) ? (int)$out : 0;
    }

    private function execString(string $cmd): ?string
    {
        if (!function_exists('shell_exec')) {
            return null;
        }
        try {
            $out = @shell_exec($cmd);
            if ($out === null) return null;
            $out = trim((string)$out);
            return $out === '' ? null : $out;
        } catch (\Throwable) {
            return null;
        }
    }

    private function pruneHistory(float $now): void
    {
        $cut = $now - $this->windowSeconds;
        while (!empty($this->history) && $this->history[0]['t'] < $cut) {
            array_shift($this->history);
        }
    }

    private function getRate(): ?float
    {
        if (count($this->history) < 2) {
            $elapsed = microtime(true) - $this->startedAt;
            if ($elapsed <= 0.0) {
                return null;
            }
            return $this->done / $elapsed;
        }
        $first = $this->history[0];
        $last = end($this->history);
        if (!$last) return null;
        $dt = $last['t'] - $first['t'];
        $dd = $last['d'] - $first['d'];
        if ($dt <= 0.0) return null;
        return max(0.0, $dd / $dt);
    }

    private function getETA(?float $rate): ?float
    {
        if ($this->total <= 0) return null;
        if ($this->done <= 0) return null;
        if ($rate === null || $rate <= 0.0) return null;
        $remaining = max(0, $this->total - $this->done);
        return $remaining / $rate;
    }

    private function render(bool $final = false): void
    {
        $elapsed = microtime(true) - $this->startedAt;
        $rate = $this->getRate();
        $eta = $this->getETA($rate);

        $percent = ($this->total > 0)
            ? min(1.0, max(0.0, $this->done / $this->total))
            : 0.0;

        // Build bar and text pieces
        $label = (string)$this->opt['label'];
        $status = (string)$this->opt['status'];
        $spinner = $this->opt['show_spinner'] ? $this->nextSpinner() : '';

        $leftPrefix = $label !== '' ? $label . ' ' : '';
        if ($spinner !== '') {
            $leftPrefix .= $spinner . ' ';
        }

        $rightBits = [];

        if ($this->opt['show_percent']) {
            $rightBits[] = sprintf('%3d%%', (int)round($percent * 100));
        }
        if ($this->opt['show_counters'] && $this->total > 0) {
            $rightBits[] = sprintf('%d/%d', $this->done, $this->total);
        } elseif ($this->opt['show_counters'] && $this->total === 0) {
            $rightBits[] = (string)$this->done;
        }
        if ($this->opt['show_rate'] && $rate !== null) {
            $rightBits[] = sprintf('%s/s', $this->formatNumber($rate));
        }
        if ($this->opt['show_elapsed']) {
            $rightBits[] = 'elapsed ' . $this->formatDuration($elapsed);
        }
        if ($this->opt['show_eta'] && $eta !== null && !$final) {
            $rightBits[] = 'eta ' . $this->formatDuration($eta);
        }

        if ($this->opt['show_memory']) {
            $mem = memory_get_usage(true);
            $rightBits[] = 'mem ' . $this->formatBytes($mem);
        }
        if ($this->opt['show_peak_memory']) {
            $peak = memory_get_peak_usage(true);
            $rightBits[] = 'peak ' . $this->formatBytes($peak);
        }

        $rightText = implode('  ', $rightBits);
        $bar = $this->buildBar($percent, $leftPrefix, $rightText, $status);

        // Move cursor and print
        $line = "\r" . $bar;
        if (!$this->isTty) {
            // For non-TTY (e.g., redirected output), print lines less frequently
            // or fall back to periodic summaries. Here we still print carriage return,
            // but some environments will treat it as newline visually.
        }
        $this->write($line);
    }

    private function buildBar(float $percent, string $leftPrefix, string $rightText, string $status): string
    {
        $ld = (string)$this->opt['left_delim'];
        $rd = (string)$this->opt['right_delim'];

        // Compute available width for bar + status
        $totalWidth = max(20, $this->termWidth);
        $fixedParts = $leftPrefix . $ld . $rd . ' ' . $rightText;
        if ($status !== '') {
            $fixedParts .= '  ' . $status;
        }

        // We'll rebuild to fit: start with ideal bar width from option, then shrink as needed.
        $barWidth = (int)$this->opt['width'];
        $overhead = strlen($leftPrefix) + strlen($ld) + strlen($rd) + 1 /*space*/ + strlen($rightText);
        $statusLen = ($status !== '') ? (2 + strlen($status)) : 0; // "  {status}"
        $needed = $overhead + $statusLen + $barWidth;

        if ($needed > $totalWidth) {
            // Shrink status first
            $spaceForStatus = max(0, $totalWidth - $overhead - $barWidth);
            $statusStr = '';
            if ($status !== '' && $spaceForStatus > 3) {
                $statusStr = '  ' . $this->truncateRight($status, $spaceForStatus - 2);
            } elseif ($status !== '') {
                $statusStr = '';
            }
            // If still too big, shrink bar
            $needed2 = $overhead + strlen($statusStr) + $barWidth;
            if ($needed2 > $totalWidth) {
                $barWidth = max(5, $barWidth - ($needed2 - $totalWidth));
            }
            $statusOut = $statusStr;
        } else {
            $statusOut = ($status !== '') ? '  ' . $status : '';
        }

        // Compute bar segments
        $filled = (int)floor($percent * $barWidth);
        $hasHead = ($percent > 0.0 && $percent < 1.0 && $barWidth >= 2);
        $empty = $barWidth - $filled - ($hasHead ? 1 : 0);

        $barDone = str_repeat((string)$this->opt['bar_char'], max(0, $filled));
        $barHead = $hasHead ? (string)$this->opt['head_char'] : '';
        $barEmpty = str_repeat((string)$this->opt['empty_char'], max(0, $empty));

        $barPlain = $ld . $barDone . $barHead . $barEmpty . $rd;

        if ($this->opt['colorize']) {
            $done = $this->color((string)$this->opt['fg_done'], $barDone);
            $head = $this->color((string)$this->opt['fg_done'], $barHead);
            $empty = $this->color((string)$this->opt['fg_empty'], $barEmpty);
            $barPlain = $ld . $done . $head . $empty . $rd;
            $leftPrefix = $this->color((string)$this->opt['fg_text'], $leftPrefix);
            $rightText = $this->color((string)$this->opt['fg_text'], $rightText);
            $statusOut = $statusOut !== '' ? $this->color((string)$this->opt['fg_text'], $statusOut) : '';
        }

        return $leftPrefix . $barPlain . ' ' . $rightText . $statusOut;
    }

    private function nextSpinner(): string
    {
        $chars = (array)$this->opt['spinner'];
        if (empty($chars)) {
            $chars = ['|', '/', '-', '\\'];
        }
        $ch = $chars[$this->spinnerIndex % count($chars)];
        $this->spinnerIndex++;
        return $this->opt['colorize'] ? $this->color((string)$this->opt['fg_text'], (string)$ch) : (string)$ch;
    }

    private function truncateRight(string $s, int $max): string
    {
        if ($max <= 0) return '';
        if (strlen($s) <= $max) return $s;
        if ($max <= 1) return '…';
        return substr($s, 0, $max - 1) . '…';
    }

    private function color(string $fgCode, string $text): string
    {
        return "\033[" . $fgCode . "m" . $text . "\033[0m";
    }

    private function write(string $s): void
    {
        // Use STDOUT explicitly
        fwrite(STDOUT, $s);
        fflush(STDOUT);
    }

    private function installSignalHandlerIfAvailable(): void
    {
        if (function_exists('pcntl_async_signals') && function_exists('pcntl_signal')) {
            @pcntl_async_signals(true);
            @pcntl_signal(SIGINT, function () {
                // Print a newline to leave the bar in a clean state.
                $this->write("\r");
                $this->finish('Interrupted');
                exit(130); // 128+SIGINT
            });
        }
    }

    /* ========================== Formatters ========================== */

    private function formatDuration(float $seconds): string
    {
        $seconds = (int)round($seconds);
        if ($seconds < 60) {
            return $seconds . 's';
        }
        $m = intdiv($seconds, 60);
        $s = $seconds % 60;
        if ($m < 60) {
            return sprintf('%dm%02ds', $m, $s);
        }
        $h = intdiv($m, 60);
        $m = $m % 60;
        return sprintf('%dh%02dm', $h, $m);
    }

    private function formatBytes(int $bytes): string
    {
        $units = ['B','KB','MB','GB','TB','PB'];
        $i = 0;
        $val = (float)$bytes;
        while ($val >= 1024 && $i < count($units) - 1) {
            $val /= 1024;
            $i++;
        }
        if ($val >= 100) {
            return sprintf('%.0f%s', $val, $units[$i]);
        }
        if ($val >= 10) {
            return sprintf('%.1f%s', $val, $units[$i]);
        }
        return sprintf('%.2f%s', $val, $units[$i]);
    }

    private function formatNumber(float $n): string
    {
        if ($n >= 100) return number_format($n, 0, '.', '');
        if ($n >= 10)  return number_format($n, 1, '.', '');
        return number_format($n, 2, '.', '');
    }
}

/* ========================== Demo (optional) ==========================
   Run: php progress.php
   Comment out or remove this block in production if you only want the class.
*/
if (PHP_SAPI === 'cli' && realpath($_SERVER['SCRIPT_FILENAME']) === __FILE__) {
    $total = 137;

    $bar = new ProgressBarCLI($total, [
        'label' => 'Processing',
        'show_peak_memory' => true,
        'width' => 50,              // hint; auto-fits if terminal is small
        'min_interval' => 0.03,     // smoother animation
        // 'colorize' => false,      // uncomment to disable colors
    ]);

    for ($i = 0; $i <= $total; $i++) {
        // Optional status per iteration
        $status = ($i % 10 === 0) ? "checkpoint #$i" : '';
        $bar->update($i, null, $status);

        // Simulate work
        usleep(random_int(3000, 20000)); // 3–20ms
    }
    $bar->finish('Done');
}