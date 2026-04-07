@echo off
if not defined VISUAL_DSA (
	call prepare.bat > nul
)

if not exist build mkdir build

pushd build

REM cl /O2 /LD ..\haneul.c /link /EXPORT:fit /EXPORT:predict /EXPORT:init_knn
REM cl /Od /Zi /LD ..\haneul.c /link /EXPORT:fit /EXPORT:predict /EXPORT:init_knn
clang-cl /D THREAD_COUNT=16 /D FEATURE_MULTITHREADED /Ox /LD ..\haneul.c /link /EXPORT:fit /EXPORT:predict /EXPORT:init_knn

REM clang-cl /Od /Zi /LD ..\haneul.c /link /EXPORT:fit /EXPORT:predict /EXPORT:init_knn /EXPORT:quickselect
REM clang-cl /Od /Zi ..\test_win32.c /Fe:test.exe

copy haneul.dll .. > nul
popd build