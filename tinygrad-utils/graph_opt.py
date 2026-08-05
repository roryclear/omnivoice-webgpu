import json


def get_buffers(file_name):
  data = json.load(open(file_name))
  buffers = set()
  copyins = set()
  for element in data:
    if "call" in element.keys():
      for b in element["call"]["buffers"]: buffers.add(b)
    elif "copyin" in element.keys():
      copyins.add(element["copyin"]["dest"])
  return buffers, copyins

print(get_buffers("1.rc"))