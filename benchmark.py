import pygame, timeit, random, math
from graph import *

# benchmarking for naive and optimized dijsktra's
# runs dijsktra's on various images (shortest path from source to all vertices)
# Images = ["sample.png","frenchbulldog.png","bear_cub.png","sleeping_fox.png","deer.png","squirrel.png",
#           "squirrel_fun.png","tiger.png"]
Images = ["cute_white_animal.png","cute_puppy_cartoon.png"]

for i in Images:
  print(i)
  IMAGE = i
  pygame.init()
  image = pygame.image.load(IMAGE)
  image_rect = image.get_rect()
  width = image_rect.width
  height = image_rect.height
  # randomly choose starting vertex
  x_start = random.randint(0, width-1)
  y_start = random.randint(0, height-1)

  graph = image_to_graph(image, image_rect.width, image_rect.height)

  start = timeit.default_timer()
  num_vertices_a = shortest_path_optimized_bench(
    graph,
    xy_to_vertex(x_start, y_start, image_rect.width)
  )
  stop = timeit.default_timer()
  optimized = stop - start
  print('Time Optimized: ', optimized)
  thoery_xfaster = (num_vertices_a*(2*width+2*height))/(math.log2(num_vertices_a)*(num_vertices_a+2*num_vertices_a-image_rect.width-image_rect.height))
  print('In Theory (xFaster): ', thoery_xfaster)
  start = timeit.default_timer()
  num_vertices_b = shortest_path_bench(
    graph,
    xy_to_vertex(x_start, y_start, image_rect.width)
  )
  stop = timeit.default_timer()
  naive = stop - start
  print('Time Naive: ',naive)
  practice_xfaster = naive/optimized
  print('In Practice (xFaster): ', practice_xfaster)
  # thoery_xfaster = pow(num_vertices_a,2)/(math.log2(num_vertices_a)*(num_vertices_a+2*num_vertices_a-image_rect.width-image_rect.height))
  # thoery_xfaster = (num_vertices_a*(2*width+2*height))/(math.log2(num_vertices_a)*(num_vertices_a+2*num_vertices_a-image_rect.width-image_rect.height))
  print('In Theory (xFaster): ', thoery_xfaster)
  print('Total Vertices: ', num_vertices_a)



