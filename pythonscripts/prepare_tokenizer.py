import sys
import pickle
import struct
import json
data = json.load(open("vocab.json"))

def write():
    with open('dictionary.bin', 'wb') as f:
        magic_num = 1337
        buffer = struct.pack('i', magic_num) 
        f.write(buffer)
        
        dict_size = len(data)
        buffer = struct.pack('i', dict_size)
        f.write(buffer)

        for key, val in data.items():
            size_key = len(key)
            bkey = key.encode('utf-8')
            sizebuffer = struct.pack('i', size_key)
            buffer = struct.pack(f'{size_key}s', bkey)
            f.write(sizebuffer)
            f.write(buffer)
            
            valbuffer = struct.pack('i', val)
            f.write(valbuffer)

    print("Dictionary written to binary file successfully.")

def read(path):
    u_dict = {}
    with open("dictionary.bin", "rb") as f:
        data = f.read()
        offset = 0
        magic_num = struct.unpack('i', data[offset:offset+4])[0]
        offset += 4
        dict_size = struct.unpack('i', data[offset:offset+4])[0]
        offset += 4
        for i in range(dict_size):
            key_size = struct.unpack('i', data[offset:offset+4])[0]
            offset += 4
            try:
                key = data[offset:offset+key_size].decode('cp1252')
            except:
                key = "NULLMF"
                offset += key_size
                val = struct.unpack('i', data[offset:offset+4])[0]
                offset += 4
                u_dict[key] = val

    print(len(u_dict))
    return u_dict

write()