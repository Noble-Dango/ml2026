clang -shared -fPIC -O2 haneul.c -o libhaneul.so \
    --target=x86_64-linux-musl \
    -I/usr/include/x86_64-linux-musl \
    -L/usr/lib/x86_64-linux-musl \
    -static
