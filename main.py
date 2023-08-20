import itertools, math, pygame, sys
import numpy as np
from graph import *

IMAGE = "sample.png"

pygame.init()

image = pygame.image.load(IMAGE)
image_edit = image.copy()

image_rect = image.get_rect()
anchor_image = pygame.image.load("anchor.png")
anchor_red_image = pygame.image.load("anchor_red.png")
clock = pygame.time.Clock()

start = timeit.default_timer()
graph = image_to_graph(image, image_rect.width, image_rect.height)
stop = timeit.default_timer()
optimized = stop - start
print('Time Image to Graph: ', optimized)

anchors = []
path = []
current_path = []
path_animation_offset = 0
path_animation_length = 10
screen = pygame.display.set_mode(image_rect.size)

# Draw an animated striped path.
def blit_path():
  global path_animation_offset, path_animation_length
  path_animation_offset = (path_animation_offset + 0.25)%path_animation_length
  for i,v in enumerate(path + current_path):
    c = 255*math.sin((i+path_animation_offset)*2*math.pi/path_animation_length)**2
    screen.set_at(v, (c,c,c))

while True:
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      sys.exit()
    if event.type == pygame.MOUSEBUTTONDOWN:
      # Add an anchor on left click
      if event.button == 1:
        path += current_path
        anchors.append(event.pos)
      # Reset anchors on right click
      if event.button == 3:
        path = []
        current_path = []
        anchors = []

  # Update path from most recent anchor to mouse.
  mouse_x, mouse_y = pygame.mouse.get_pos()
  if len(anchors) > 0:
    anchor_x, anchor_y = anchors[-1]
    current_path = []
    # # only for visualizations
    # shortest_path, image_edit = shortest_path_optimized_vis(
    # graph,
    # xy_to_vertex(anchor_x, anchor_y, image_rect.width),
    # xy_to_vertex(mouse_x, mouse_y, image_rect.width),
    # image)
    shortest_path = shortest_path_optimized(
    graph,
    xy_to_vertex(anchor_x, anchor_y, image_rect.width),
    xy_to_vertex(mouse_x, mouse_y, image_rect.width))
    for v in shortest_path:
      current_path.append(vertex_to_xy(v, image_rect.width))

  # Blit images to screen.
  screen.blit(image_edit, image_rect)
  blit_path()

  for anchor in anchors:
    screen.blit(anchor_image if anchor != anchors[-1] else anchor_red_image, anchor_image.get_rect(center=anchor))

  pygame.display.flip()
  clock.tick(60)
