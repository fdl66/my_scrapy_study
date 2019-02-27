#!/usr/bin/python
import json

def main():
    dest = list()
    with open("./dests.json", "r") as f_obj:
        dest = json.loads(f_obj.read())
    for i in dest:
        print(i)

if __name__ == "__main__":
    import pdb;pdb.set_trace()
    main()
