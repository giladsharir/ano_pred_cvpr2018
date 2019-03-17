
def get_kinetics_names(ntu=False):
  if ntu:
      class_file = 'ntu_classes.txt'
  else:
      class_file = 'kinetics_classes.txt'
  class_names = []

  with open(class_file) as f:
    for l in f.readlines():
      class_names.append(l.split(',')[1][2:-2])

  return class_names
