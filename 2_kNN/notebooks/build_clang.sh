clang -shared -fPIC -O2 inplace_argsort.c -o libinplace_argsort.so \
    --target=x86_64-linux-musl \
    -I/usr/include/x86_64-linux-musl \
    -L/usr/lib/x86_64-linux-musl \
    -static
