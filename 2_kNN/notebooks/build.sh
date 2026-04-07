gcc -D FEATURE_MULTITHREADED -shared -fPIC -fno-strict-aliasing -O2 haneul.c -o libhaneul.so

# musl-gcc -D FEATURE_MULTITHREADED -static -shared -fPIC -fno-strict-aliasing -O2 haneul.c -o libhaneul.so

# clang -D FEATURE_MULTITHREADED -shared -fPIC -fno-strict-aliasing -O2 haneul.c -o libhaneul.so

# clang -D THREAD_COUNT=8  -shared -fno-strict-aliasing -fPIC -O2 haneul.c -o libhaneul.so \
#   --target=x86_64-linux-musl \
#    -I/usr/include/x86_64-linux-musl \
#    -L/usr/lib/x86_64-linux-musl \
#    -static 
