import pyglet
from fireplace import game
from fireplace.cards import CardType
from math import *
from pyglet.gl import *
import numpy as np


class EntityDisplay:
	def __init__(self, name, health, attack, mana, posX, posY):
		self.name = pyglet.text.Label(name,
			                          font_name='Times New Roman',
			                          font_size=10,
			                          x=posX, y=posY,
			                          anchor_x='center', anchor_y='center')
		self.health = pyglet.text.Label(str(health),
				                          font_name='Times New Roman',
				                          font_size=10,
				                          x=posX+10, y=posY-20,
				                          anchor_x='center', anchor_y='center')
		self.atk = pyglet.text.Label(str(attack),
			                          font_name='Times New Roman',
			                          font_size=10,
			                          x=posX-10, y=posY-20,
			                          anchor_x='center', anchor_y='center')
		self.mana = pyglet.text.Label(str(mana),
			                          font_name='Times New Roman',
			                          font_size=10,
			                          x=posX+10, y=posY+20,
			                          anchor_x='center', anchor_y='center')
	def draw(self):
		self.name.draw()
		self.health.draw()
		self.atk.draw()
		if self.mana.text is not -1:
			self.mana.draw()

	def updateStats(self, name, health, atk, mana):
		if name is not None:
			self.name.text = name
		else:
			self.name.text = ""

		if health is not -1:
			self.health.text = str(health)
		else:
			self.health.text = ""

		if atk is not -1:
			self.atk.text = str(atk)
		else:
			self.atk.text = ""

		if mana is not -1:
			self.mana.text = str(mana)
		else:
			self.mana.text = ""


class BoardRenderer:
	def __init__(self):
		"""TODO: Initialize Window"""
		self.window = pyglet.window.Window(width=1200, height=600, caption="HS_AI")
		self.f_hero = EntityDisplay('Hero1', 0, 30, -1, (self.window.width // 10) * 5, (self.window.height // 10) * 2.5)
		self.o_hero = EntityDisplay('Hero2', 0, 30, -1, (self.window.width // 10) *5, (self.window.height // 10) * 7.5)
		self.f_hand = []
		self.o_hand = []
		for i in range(10):
			self.f_hand.append(EntityDisplay('Card' + str(i), 0, 0, 0,
			                                 (self.window.width // 11) * (i+1),
			                                 (self.window.height // 10) * 1))
		for i in range(10):
			self.o_hand.append(EntityDisplay('Card' + str(i), 0, 0, 0,
			                                 (self.window.width // 11) * (i+1),
			                                 (self.window.height // 10) * 9))

		self.f_field = []
		for i in range(7):
			self.f_field.append(EntityDisplay('Mninon' + str(i), 0, 0, -1,
			                                  (self.window.width // 7) * (i),
			                                  (self.window.height // 10) * 4))

		self.o_field = []
		for i in range(7):
			self.o_field.append(EntityDisplay('Mninon' + str(i), 0, 0, -1,
			                                  (self.window.width // 7) * (i),
			                                  (self.window.height // 10) * 6))


	def upadate_board(self, game):
		self.f_hero.updateStats(str(game.player1.hero), game.player1.hero.health, game.player1.hero.atk, -1)
		self.o_hero.updateStats(str(game.player2.hero), game.player2.hero.health, game.player2.hero.atk, -1)

		for i in range(10):
			self.f_hand[i].updateStats(None, -1, -1, -1)
		for i in range(10):
			self.o_hand[i].updateStats(None, -1, -1, -1)

		index = 0
		for card in game.player1.hand:
			if card.data.type is (CardType.MINION or CardType.WEAPON):
				self.f_hand[index].updateStats(str(card), card.health, card.atk, card.cost)
			else:
				self.f_hand[index].updateStats(str(card), -1, -1, card.cost)
			index += 1

		index = 0
		for card in game.player2.hand:
			if card.data.type is (CardType.MINION or CardType.WEAPON):
				self.o_hand[index].updateStats(str(card), card.health, card.atk, card.cost)
			else:
				self.o_hand[index].updateStats(str(card), -1, -1, card.cost)
			index += 1

		for i in range(7):
			self.f_field[i].updateStats(None, -1, -1, -1)
		for i in range(7):
			self.o_field[i].updateStats(None, -1, -1, -1)

		index = 0
		for card in game.player1.field:
			self.f_field[index].updateStats(str(card), card.health, card.atk, -1)
			index += 1

		index = 0
		for card in game.player2.field:
			self.o_field[index].updateStats(str(card), card.health, card.atk, -1)
			index += 1

	def render_board_state(self, game):
		"""TODO: Render Boardstate"""
		self.upadate_board(game)

		self.window.switch_to()
		self.window.clear()
		self.f_hero.draw()
		self.o_hero.draw()
		for card in self.f_hand:
			card.draw()
		for card in self.o_hand:
			card.draw()
		for minion in self.f_field:
			minion.draw()
		for minion in self.o_field:
			minion.draw()

		pyglet.clock.tick()

		self.window.dispatch_event('on_draw')

		for window in pyglet.app.windows:
			window._allow_dispatch_event = True
		self.window.dispatch_events()
		for window in pyglet.app.windows:
			window._allow_dispatch_event = False
		self.window.flip()




class Circle:

	def __init__(self, sizeX, sizeY, numPoints, posX, posY):
		self.verts = []
		self.colors = []
		self.numPoints = numPoints
		for i in range(self.numPoints):
			angle = radians(float(i)/self.numPoints * 360.0)
			x = sizeX*cos(angle)
			y = sizeY*sin(angle)
			self.verts += [x+posX,y+posY]
			self.colors += [255, 255, 255]
		global circle
		self.circle = pyglet.graphics.vertex_list(self.numPoints, ('v2f', self.verts),
		                                          ('c3B', self.colors))
	def draw(self, color):
		self.colors.clear()
		for i in range(self.numPoints):
			self.colors += [0, int(max(0, color)*200), 0]
		self.circle = pyglet.graphics.vertex_list(self.numPoints, ('v2f', self.verts),
                                                    ('c3B', self.colors))
		self.circle.draw(GL_LINE_LOOP)


class Node:
	def __init__(self, size, posX, posY):
		self.pos = (posX, posY)
		self.circle = Circle(sizeX=size*4, sizeY=size, numPoints=100, posX=posX, posY=posY)
		if size > 2:
			self.label = pyglet.text.Label("",
				                          font_name='Times New Roman',
				                          font_size=min(size*2, 12),
				                          x=posX, y=posY,
				                          anchor_x='center', anchor_y='center')
		else:
			self.label = None
	def update(self, value):
		self.color = value
		if self.label is not None:
			self.label.text = str(value)
	def draw(self):
		self.circle.draw(self.color)
		if self.label is not None:
			self.label.draw()


class Connection:
	def __init__(self, node_in, node_out):
		self.start_pos = node_in.pos
		self.end_pos = node_out.pos
		self.weight = 0.1
	def update(self, weight):
		self.weight = weight
	def draw(self):
		pyglet.graphics.glLineWidth(self.weight)
		pyglet.graphics.draw(2, GL_LINE_STRIP, ('v2f', (self.start_pos[0], self.start_pos[1],
	                                                    self.end_pos[0], self.end_pos[1])))

class Layer:
	def __init__(self, env, network, name, posX, windowHeight):
		state = np.reshape(env.create_board_state(), (110))
		layer_tensor = network.get_layer_tensor(layer_name=name)
		values = network.get_tensor_value(tensor=layer_tensor, state=state)[0]
		size = len(values)
		self.name = name
		self.nodes = []
		for i in range(size):
			radius = ((windowHeight /100) * 90) /size
			self.nodes.append(Node(size=radius/2, posX=posX, posY=((size-i) * radius) + ((windowHeight /100) * 5)))
	def update(self, env, network):
		state = np.reshape(env.create_board_state(), (110))
		layer_tensor = network.get_layer_tensor(layer_name=self.name)
		values = network.get_tensor_value(tensor=layer_tensor, state=state)[0]
		for i in range(len(values)):
			self.nodes[i].update(values[i])
	def draw(self):
		for node in self.nodes:
			node.draw()

class InLayer:
	def __init__(self, env, posX, windowHeight):
		self.nodes = []
		state = env.create_board_state()
		size = len(state)
		for i in range(size):
			radius = ((windowHeight / 100) * 90) / size
			self.nodes.append(Node(size=radius / 2, posX=posX, posY=((size-i) * radius) + ((windowHeight / 100) * 5)))

	def update(self, env):
		state = env.create_board_state()
		size = len(state)
		for i in range(size):
			self.nodes[i].update(state[i])

	def draw(self):
		for node in self.nodes:
			node.draw()

class OutLayer:
	def __init__(self, env, network, posX, windowHeight):
		self.nodes = []
		state = np.reshape(env.create_board_state(), (110))
		values = network.get_q_values(states=[state])[0]
		size = len(values)
		for i in range(size):
			radius = ((windowHeight / 100) * 90) / size
			self.nodes.append(Node(size=radius / 2, posX=posX, posY=((size-i) * radius) + ((windowHeight / 100) * 5)))

	def update(self, env, network):
		state = np.reshape(env.create_board_state(), (110))
		values = network.get_q_values(states=[state])[0]
		size = len(values)
		for i in range(size):
			self.nodes[i].update(values[i])

	def draw(self):
		for node in self.nodes:
			node.draw()

class ChoicesDisplay:
	def __init__(self, env, posX, windowHeight):
		self.choices = []
		meanings = env.get_action_meanings()
		size = len(meanings)
		radius = ((windowHeight / 100) * 90) / size
		for i in range(size):
			label = pyglet.text.Label(meanings[i],
			                          font_name='Times New Roman',
			                          font_size=12,
			                          x=posX, y=((size-i) * radius) + ((windowHeight /100) * 5),
									  anchor_x='center', anchor_y='center')
			self.choices.append(label)
	def draw(self):
		for label in self.choices:
			label.draw()

class Connections:
	def __init__(self, layer_in, layer_out):
		self.connections = []
		for i in layer_in.nodes:
			for j in layer_out.nodes:
				self.connections.append(Connection(i, j))
	def update(self, weights):
		return
	def draw(self):
		for conn in self.connections:
			conn.draw()


class NetworkRenderer:
	def __init__(self, network, env, num_layers, layers_sizes):
		"""TODO: Initialize Window"""
		self.window = pyglet.window.Window(width=1300, height=1000, caption="Network Display")

		self.in_layer = InLayer(env=env, posX=100.0, windowHeight=self.window.height)
		self.in_layer.update(env=env)

		self.fc_layer_1 = Layer(env=env, network=network, name='layer_fc1_a', posX=200.0, windowHeight=self.window.height)
		self.fc_layer_1.update(env=env, network=network)

		self.fc_layer_2 = Layer(env=env, network=network, name='layer_fc2_a', posX=300.0, windowHeight=self.window.height)
		self.fc_layer_2.update(env=env, network=network)

		self.fc_layer_3 = Layer(env=env, network=network, name='layer_fc3_a', posX=400.0, windowHeight=self.window.height)
		self.fc_layer_3.update(env=env, network=network)

		self.fc_layer_4 = Layer(env=env, network=network, name='layer_fc4_a', posX=500.0, windowHeight=self.window.height)
		self.fc_layer_4.update(env=env, network=network)

		self.fc_layer_5 = Layer(env=env, network=network, name='layer_fc5_a', posX=600.0, windowHeight=self.window.height)
		self.fc_layer_5.update(env=env, network=network)

		self.fc_layer_6 = Layer(env=env, network=network, name='layer_fc6_a', posX=700.0, windowHeight=self.window.height)
		self.fc_layer_6.update(env=env, network=network)

		self.fc_layer_7 = Layer(env=env, network=network, name='layer_fc7_a', posX=800.0, windowHeight=self.window.height)
		self.fc_layer_7.update(env=env, network=network)

		self.fc_layer_8 = Layer(env=env, network=network, name='layer_fc8_a', posX=900.0, windowHeight=self.window.height)
		self.fc_layer_8.update(env=env, network=network)

		self.out_layer = OutLayer(env=env, network=network, posX=1000.0, windowHeight=self.window.height)
		self.out_layer.update(env=env, network=network)

		self.choices_labels = ChoicesDisplay(env=env, posX=1200.0, windowHeight=self.window.height)

		#self.out_layer = Layer(env=env, network=network, name='layer_fc_out_a', posX=1000.0, windowHeight=self.window.height)
		#self.out_layer.update(env=env, network=network)
		# self.conn = Connections(self.in_layer, self.out_layer)

	def update(self, network, env):
		self.in_layer.update(env=env)
		self.fc_layer_1.update(env=env, network=network)
		self.fc_layer_2.update(env=env, network=network)
		self.fc_layer_3.update(env=env, network=network)
		self.fc_layer_4.update(env=env, network=network)
		self.fc_layer_5.update(env=env, network=network)
		self.fc_layer_6.update(env=env, network=network)
		self.fc_layer_7.update(env=env, network=network)
		self.fc_layer_8.update(env=env, network=network)
		self.out_layer.update(env=env, network=network)

	def draw(self):
		# pyglet.clock.tick()

		self.window.switch_to()
		self.window.clear()

		self.in_layer.draw()
		self.fc_layer_1.draw()
		self.fc_layer_2.draw()
		self.fc_layer_3.draw()
		self.fc_layer_4.draw()
		self.fc_layer_5.draw()
		self.fc_layer_6.draw()
		self.fc_layer_7.draw()
		self.fc_layer_8.draw()
		self.out_layer.draw()
		self.choices_labels.draw()

		# self.conn.draw()

		pyglet.clock.tick()
		self.window.dispatch_event('on_draw')
		for window in pyglet.app.windows:
			window._allow_dispatch_event = True
		self.window.dispatch_events()
		for window in pyglet.app.windows:
			window._allow_dispatch_event = False
		self.window.flip()

