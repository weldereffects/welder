# Welder ML setup: which GPU to install PyTorch for. Prints "<kind>|<the display adapters Windows lists>",
# kind being nvidia, amd, intel or cpu. NVIDIA first (CUDA runs everything), then a discrete Radeon, then
# Intel Arc (discrete, or Core Ultra graphics), then any Radeon; older Intel graphics are slower than the CPU.
$names = @()
try { $names = @(Get-CimInstance Win32_VideoController -ErrorAction Stop | ForEach-Object { $_.Name }) }
catch { try { $names = @(Get-WmiObject Win32_VideoController | ForEach-Object { $_.Name }) } catch { } }
$all = ($names | Where-Object { $_ }) -join '; '
if ($all -match 'NVIDIA') { $kind = 'nvidia' }
elseif ($all -match 'Radeon(\(TM\))? (RX|Pro|VII)') { $kind = 'amd' }
elseif ($all -match 'Arc') { $kind = 'intel' }
elseif ($all -match 'Radeon') { $kind = 'amd' }
else { $kind = 'cpu' }
Write-Output "$kind|$all"
