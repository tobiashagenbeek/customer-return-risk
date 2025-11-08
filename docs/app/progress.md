# Progress Bar Utility Documentation 

## Overview 

The ProgressBarCLI class is a PHP CLI utility designed to provide real-time 
progress feedback during long-running operations. It supports features such 
as ETA calculation, elapsed time, memory usage, spinner animation, and ANSI 
color formatting. This utility is intended for use in terminal environments 
and is highly customizable through a variety of options. 

## Key Features 

- Works in PHP CLI only 
- Provides ETA, elapsed time, rate, memory usage, ANSI colors, spinner, and 
auto-fit width 
- Customizable display options including percent, counters, elapsed time, ETA, rate, 
memory, and spinner 
- Supports signal handling for graceful interruption 
- Colorized output with configurable ANSI codes 
- Auto-detects terminal width and adapts bar size accordingly 

## Usage 

To use the ProgressBarCLI class, instantiate it with the total number of items 
and optional configuration parameters. Call update() to refresh the progress bar, 
and finish() when the task is complete. 

## Example 

```php
$bar = new ProgressBarCLI($total, [ 
    'label' => 'Processing', 
    'show_peak_memory' => true, 
    'width' => 50, 
    'min_interval' => 0.03 
]); 
for ($i = 0; $i <= $total; $i++) { 
    $bar->update($i); 
    usleep(10000); 
} 
$bar->finish('Done'); 
```

## Integration Notes

This utility is integrated into the data generation process via the 
DataGeneratorCLI class in `data.php`. It provides visual feedback during the 
creation of customer data records. Ensure that Progress.php is included 
using require_once and that the script is executed in CLI mode. 
