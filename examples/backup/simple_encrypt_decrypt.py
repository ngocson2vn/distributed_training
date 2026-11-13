#!/usr/bin/python

import sys
import os
import struct
from getopt import getopt

fnames = ["cl_model.tar"]

def simple_encrypt(key):
	for fn in fnames:
		print("Encrypting file: {}".format(fn))
		reader = open(fn, "rb")
		writer = open(fn + ".encrypted", "wb")
		data = [x for x in bytearray(reader.read())]
		xored = [x ^ key for x in data]
		packed = [struct.pack("i", x) for x in xored]
		writer.write(b''.join(packed))
		reader.close()
		writer.close()

def simple_decrypt(key):
	for fn in fnames:
		print("Decrypting file: {}".format(fn))
		reader = open(fn + ".encrypted", "rb")
		writer = open(fn, "wb")
		data = reader.read()
		unpacked = []
		i = 0
		while i < len(data):
			k = data[i: i + 4]
			unpacked.append(struct.unpack("i", k)[0])
			i += 4
		xored = [x ^ key for x in unpacked]
		writer.write(bytearray(xored))
		reader.close()
		writer.close()

options, _ = getopt(sys.argv[1:], "edk:", ["encrypt", "decrypt"])
is_encryption = False
is_decryption = False
key = None
for opt, value in options:
	if opt == "-e":
		is_encryption = True
	if opt == "-d":
		is_decryption = True
	if opt == "-k":
		key = int(value)

if is_encryption:
	simple_encrypt(key)
	print("Encrypted files: {}".format(fnames))
elif is_decryption:
	simple_decrypt(key)
	print("Decrypted files: {}".format(fnames))