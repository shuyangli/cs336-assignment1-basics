## unicode1

a. `chr(0)` returns the Null unicode character.
b. The string representation is a 0 byte (\x00), and the printed representation is the empty string.
c. In text it functions the same way: string representation is a 0 byte, and printed representation is empty string.

## unicode2

a. UTF-8 uses fewer bytes per character; UTF-16 and UTF-32 uses a minimum of 2/4 bytes per character, and UTF-8 uses only one. This means our vocabulary size for UTF-8 encoded strings can be smaller.
b. The functions decodes each byte one by one, but not each character is one byte in UTF-8. Any character taking more than 1 byte (e.g. a Chinese character) will break that function.
c. '\xff\x01': byte 0 doesn't follow the expected encoding.