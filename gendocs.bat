pdoc -d numpy -o docs eckity
::SET src_folder=docs\eckity
::SET dst_folder=docs
::for /f %%a IN ('dir "%src_folder%" /b') do move "%src_folder%\%%a" "%dst_folder%\"
::rmdir %src_folder%