import json


def get_buffers(file_name):
  data = json.load(open(file_name))
  buffers = set()
  for element in data:
    if "call" in element.keys():
      for b in element["call"]["buffers"]: buffers.add(b)
  return buffers

def opt(file_a, file_b):
  a_bufs = get_buffers(file_a)

  data_a = json.load(open(file_a))
  data_b = json.load(open(file_b))

  for element in data_a:
    if "copyin" in element.keys() and element["copyin"]["dest"] not in a_bufs:
      data_b.insert(0, element)
      data_a.remove(element) # todo this is slow
  for element in data_a:
    if "buff_alloc" in element.keys() and element["buff_alloc"]["num"] not in a_bufs:
      data_b.insert(0, element)
      data_a.remove(element) # todo this is slow
  
  with open(file_a, "w") as f: json.dump(data_a, f)
  with open(file_b, "w") as f: json.dump(data_b, f)

print("here")
opt("0.rc", "1.rc")
print("done")

