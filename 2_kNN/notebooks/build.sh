clang -shared -fno-strict-aliasing -fPIC -O2 haneul.c -o libhaneul.so \
    --target=x86_64-linux-musl \
    -I/usr/include/x86_64-linux-musl \
    -L/usr/lib/x86_64-linux-musl \
    -static

# musl-gcc -static -shared -fPIC -fno-strict-aliasing -O2 haneul.c -o libhaneul.so