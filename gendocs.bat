pdoc -d numpy -o doc eckity
::SET src_folder=doc\eckity
::SET dst_folder=doc
::for /f %%a IN ('dir "%src_folder%" /b') do move "%src_folder%\%%a" "%dst_folder%\"
::rmdir %src_folder%