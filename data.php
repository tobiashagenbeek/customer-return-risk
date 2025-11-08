<?php

require_once('Progress.php');

class DataGeneratorCLI
{

    private $memoryLimit;

    private $filename;
    private $numbers = [
        "total"    => 0,
        "yes"      => 0,
        "no"       => 0,
        "longest"  => 0,
        "bad_guys" => 0
    ];
    private $num = 500000;
    private $maxHistoryLength = 150;
    private $dateOffsetDays = 365 * 10;
    private $badGuyReset = 600;
    private $badGuyTarget = 99; // random seed, so

    public function __construct($filename = null, $memoryLimit = '128M', $numberOfCustomers = 500000)
    {
        if (!$filename) {
            $filename = $this->promptFilename();
        }
        $this->filename = $filename;

        $this->memoryLimit = $memoryLimit;

        ini_set('memory_limit', $this->memoryLimit);

        $this->num = $numberOfCustomers;
    }

    private function promptFilename()
    {
        echo "Enter output CSV filename: ";
        $handle = fopen("php://stdin", "r");
        $filename = trim(fgets($handle));
        fclose($handle);
        return $filename ?: 'output.csv';
    }

    public function run()
    {
        $fp = fopen($this->filename, 'w');
        fputcsv($fp, ["datetime", "returned", "number"]);

        $strTime = strtotime("-" . $this->dateOffsetDays . " days");
        $date = new \DateTime(date("Y/m/d H:i:s", $strTime));
        $badGuy = 0;
        $startPercent = 1;

        $bar = new ProgressBarCLI($this->num, [
            'label' => 'Creating Data',
            'show_peak_memory' => true,
        ]);

        for ($i = 1; $i <= $this->num; $i++) {
            $msg = [];
            $badGuy++;
            $treatBad = ($badGuy == $this->badGuyTarget);

            if ($badGuy >= $this->badGuyReset) {
                $badGuy = 0;
            }

            $msg[] = "Running Customer: $i";
            $maxHistoryLength = $this->maxHistoryLength - ($this->maxHistoryLength / $this->num) * $i;
            $msg[] = "startRand: $maxHistoryLength";
            $slant = ($maxHistoryLength / $this->num);
            $startPercent += $slant;
            $length = rand(1, $maxHistoryLength);

            if ($length > $this->numbers["longest"]) {
                $this->numbers["longest"] = $length;
            }

            for ($l = 0; $l < $length; $l++) {
                $secDiff = rand(0, 30);
                $date->modify('+' . $secDiff . ' second');
                $msg[] = "Date: " . $date->format("Y/m/d H:i:s");
                $r = rand(0, 100);

                if ($treatBad) {
                    $r = rand(50, 100);
                    $this->numbers["bad_guys"]++;
                }

                $yesOrNo = $r > $startPercent ? "no" : "yes";
                if ($yesOrNo == "yes") {
                    $msg[] = 'Using ' . $r . " // " . $yesOrNo;
                    $this->numbers["yes"]++;
                } else {
                    $this->numbers["no"]++;
                }

                $d = [
                    $date->format("Y/m/d H:i:s"),
                    $yesOrNo,
                    $i
                ];
                fputcsv($fp, $d);
                $this->numbers["total"]++;
            }
            // Implode and pass the message array to the progress bar
            $bar->update($i, $this->num, implode(", ", $msg));
        }
        $bar->finish('Created!');
        fclose($fp);

        var_dump("Numbers", $this->numbers);
    }
}

// CLI entry point
if (php_sapi_name() === 'cli') {
    $filename = $argv[1] ?? null;
    $memoryLimit = $argv[2] ?? '128M'; // Allow override from CLI
    $numberOfCustomers = $argv[3] ?? 500000; // Allow override from CLI
    $generator = new DataGeneratorCLI($filename, $memoryLimit, $numberOfCustomers);
    $generator->run();
}