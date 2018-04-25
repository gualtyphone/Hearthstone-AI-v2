#!/usr/bin/env python
import sys
import random
from fireplace import cards, game, player
from fireplace.exceptions import GameOver
from fireplace.card import Card
from hearthstone.enums import CardClass, CardType
from renderer import BoardRenderer
import time

sys.path.append("..")
THE_COIN = "GAME_005"

class HSEnv:

	def __init__(self, render):
		cards.db.initialize()
		# Create a game with the standardized decks
		if render:
			self.renderer = BoardRenderer()

		self.reset()

	def start_game(self):
		self.game.start()

		# Play the game

		"""TODO: Implement Networked version of mulligan"""
		for player in self.game.players:
			print("Can mulligan %r" % (player.choice.cards))
			mull_count = random.randint(0, len(player.choice.cards))
			cards_to_mulligan = random.sample(player.choice.cards, mull_count)
			player.choice.choose(*cards_to_mulligan)

	def draft_mage(self):
		deck = []
		deck.append(Card("EX1_277").id) #Arcane Missiles
		deck.append(Card("EX1_277").id) #Arcane Missiles
		deck.append(Card("NEW1_012").id) #Mana Wyrm
		deck.append(Card("NEW1_012").id) #Mana Wyrm
		deck.append(Card("CS2_027").id) #Mirror Image
		deck.append(Card("CS2_027").id) #Mirror Image
		deck.append(Card("EX1_066").id) #Acidic Swamp Ooze
		deck.append(Card("EX1_066").id) #Acidic Swamp Ooze
		deck.append(Card("CS2_025").id) #Arcane Explosion
		deck.append(Card("CS2_025").id) #Arcane Explosion
		deck.append(Card("CS2_024").id) #Frostbolt
		deck.append(Card("CS2_024").id) #Frostbolt
		deck.append(Card("CS2_142").id) #Kobold Geomancer
		deck.append(Card("CS2_142").id) #Kobold Geomancer
		deck.append(Card("EX1_608").id) #Sorcerer's Apprentice
		deck.append(Card("EX1_608").id) #Sorcerer's Apprentice
		deck.append(Card("CS2_023").id) #Arcane Intellect
		deck.append(Card("CS2_023").id) #Arcane Intellect
		deck.append(Card("CS2_029").id) #Fireball
		deck.append(Card("CS2_029").id) #Fireball
		deck.append(Card("CS2_022").id) #Polymorph
		deck.append(Card("CS2_022").id) #Polymorph
		deck.append(Card("CS2_033").id) #Water Elemental
		deck.append(Card("CS2_033").id) #Water Elemental
		deck.append(Card("CS2_155").id) #Archmage
		deck.append(Card("CS2_155").id) #Archmage
		deck.append(Card("EX1_559").id) #Archmage Antonidas
		deck.append(Card("CS2_032").id) #Flamestrike
		deck.append(Card("CS2_032").id) #Flamestrike
		deck.append(Card("EX1_279").id) #Pyroblast
		return deck

	def create_board_state(self):
		boardstate = []
		# Player 1 mana
		boardstate.append(self.game.player1.max_mana - self.game.player1.overload_locked)
		# Player 2 mana
		boardstate.append(self.game.player2.max_mana - self.game.player2.overload_locked)

		# Player 1 handSize
		boardstate.append(self.game.player1.hand.__len__())
		# Player 2 handSize
		boardstate.append(self.game.player2.hand.__len__())

		# Player 2 hero health
		boardstate.append(self.game.player1.hero.health + self.game.player2.hero.armor)
		# Player 2 Hero Attack value
		if self.game.player2.weapon is not None:
			boardstate.append(self.game.player2.hero.atk + self.game.player2.weapon.atk)
		else:
			boardstate.append(self.game.player2.hero.atk)

		# Player 1 hero health
		boardstate.append(self.game.player1.hero.health + self.game.player2.hero.armor)

		# Player 1 Weapon
		if self.game.player1.weapon is not None:
			# Player 1 Weapon Health
			boardstate.append(self.game.player1.weapon.health_attribute)
			# Player 1 Weapon Attack
			boardstate.append(self.game.player1.weapon.atk)
		else:
			boardstate.append(0)
			boardstate.append(0)

		# Player 2 Weapon
		if self.game.player2.weapon is not None:
			# Player 2 Weapon Health
			boardstate.append(self.game.player2.weapon.health_attribute)
			# Player 2 Weapon Attack
			boardstate.append(self.game.player2.weapon.atk)
		else:
			boardstate.append(0)
			boardstate.append(0)

		# Player 2 HAND
		for index in range(10):
			if (self.game.player2.hand.__len__() > index):
				# Card ID
				boardstate.append(sorted(cards.db).index(self.game.player2.hand[index].id))
				# Card Type
				boardstate.append(int(self.game.player2.hand[index].data.type))
				# Mana Cost
				boardstate.append(self.game.player2.hand[index].data.cost)
				if self.game.player2.hand[index].data.type is (CardType.MINION or CardType.WEAPON):
					# Card ATK
					boardstate.append(self.game.player2.hand[index].data.atk)
					# Card HEALTH
					boardstate.append(self.game.player2.hand[index].data.health)
				else:
					boardstate.append(0)
					boardstate.append(0)
			else:
				for i in range(5):
					boardstate.append(0)

		# Boardtate Player 2 Side
		for index in range(7):
			if (self.game.player2.field.__len__() > index):
				# Card ID
				boardstate.append(sorted(cards.db).index(self.game.player2.field[index].id))
				# Minion ATK
				boardstate.append(self.game.player2.field[index].atk)
				# Minion Health
				boardstate.append(self.game.player2.field[index].health)
				# Minion CanAttack
				if self.game.player2.field[index].can_attack():
					boardstate.append(1)
				else:
					boardstate.append(0)
			else:
				for i in range(4):
					boardstate.append(0)

		# Boardtate Player 1 Side
		for index in range(7):
			if (self.game.player1.field.__len__() > index):
				# Card ID
				boardstate.append(sorted(cards.db).index(self.game.player1.field[index].id))
				# Minion ATK
				boardstate.append(self.game.player1.field[index].atk)
				# Minion Health
				boardstate.append(self.game.player1.field[index].health)
			else:
				for i in range(3):
					boardstate.append(0)

		return boardstate

	def play_turn_random(self):
		player = self.game.current_player

		while True:
			heropower = player.hero.power
			if heropower.is_usable() and random.random() < 0.1:
				if heropower.requires_target():
					heropower.use(target=random.choice(heropower.targets))
				else:
					heropower.use()
				continue

			# iterate over our hand and play whatever is playable
			for card in player.hand:
				if card.is_playable() and random.random() < 0.1:
					target = None
					if card.must_choose_one:
						card = random.choice(card.choose_cards)
					if card.requires_target():
						target = random.choice(card.targets)
					print("Playing %r on %r" % (card, target))
					card.play(target=target)

					if player.choice:
						choice = random.choice(player.choice.cards)
						print("Choosing card %r" % (choice))
						player.choice.choose(choice)

					continue

			# Randomly attack with whatever can attack
			for character in player.characters:
				if character.can_attack():
					character.attack(random.choice(character.targets))

			break

		self.game.end_turn()
		return game

	def play_turn_untrained_network(self):
		while True:
			action = random.randrange(0, 34, 1)

			if action is not None:
				player = self.game.current_player
				if action is 0:
					self.game.end_turn()
					return
				elif action is 1:
					if player.hero.can_attack():
						player.hero.attack(target=random.choice(player.hero.targets))
				elif action is 2:
					heropower = player.hero.power
					if heropower.is_usable():
						if heropower.requires_target():
							heropower.use(target=random.choice(heropower.targets))
						else:
							heropower.use()
				elif action is 3:
					# find the coin and play it
					for card in player.hand:
						if card.data.id == THE_COIN and card.is_playable():
							card.play()
				else:
					to_play = self.draft_mage()[action-4]
					for card in player.hand:
						if card.id == to_play and card.is_playable():
							target = None
							if card.must_choose_one:
								card = random.choice(card.choose_cards)
							if card.requires_target():
								target = random.choice(card.targets)
							print("Playing %r on %r" % (card, target))
							card.play(target=target)

							if player.choice:
								choice = random.choice(player.choice.cards)
								print("Choosing card %r" % (choice))
								player.choice.choose(choice)
							break

					for minion in player.field:
						if minion.id == to_play and minion.can_attack():
							target = None
							minion.attack(target=random.choice(minion.attack_targets))
			if self.check_remaining_actions():
				return

	def check_remaining_actions(self):
		player = self.game.current_player

		# check if there are any more possible actions
		heropower = player.hero.power
		if heropower.is_usable():
			return False

		# iterate over our hand and play whatever is playable
		for card in player.hand:
			if card.is_playable():
				return False

		# Randomly attack with whatever can attack
		for character in player.characters:
			if character.can_attack():
				return False

		# NO MORE ACTIONS POSSIBLE - end turn
		self.game.end_turn()
		return True

	def get_action_meanings(self):
		strings = list()
		strings.append("END_TURN")
		strings.append("HERO")
		strings.append("HERO_POWER")
		strings.append("THE_COIN")
		for card in self.draft_mage():
			strings.append(Card(card).data.name)

		return strings

	def reset(self):
		from fireplace.game import Game
		from fireplace.player import Player

		deck1 = self.draft_mage()
		deck2 = self.draft_mage()
		player1 = Player("Player1", deck1, CardClass.MAGE.default_hero)
		player2 = Player("Player2", deck2, CardClass.MAGE.default_hero)

		self.game = Game(players=(player1, player2))
		self.start_game()

		self.new_game = True

		return self.create_board_state()

	def step(self, action, targeting_agent, learn_to_play_cards):
		if self.game.ended:
			if learn_to_play_cards:
				return self.create_board_state(), 0, self.is_game_over()
			else:
				return self.create_board_state(), self.calculate_reward(), self.is_game_over()
		if self.game.current_player is not self.game.player2:
			self.play_turn_untrained_network()
			if learn_to_play_cards:
				return self.create_board_state(), 0, self.is_game_over()
			else:
				return self.create_board_state(), self.calculate_reward(), self.is_game_over()

		if action is not None:
			player = self.game.current_player
			if action is 0:
				self.game.end_turn()
				if learn_to_play_cards:
					return self.create_board_state(), 0, self.is_game_over()
				else:
					return self.create_board_state(), self.calculate_reward(), self.is_game_over()
			elif action is 1:
				if player.hero.can_attack():
					targeting_agent.run(action, self.new_game)
					self.new_game = False
					self.check_remaining_actions()
					if learn_to_play_cards:
						return self.create_board_state(), 0, self.is_game_over()
					else:
						return self.create_board_state(), self.calculate_reward(), self.is_game_over()
				if learn_to_play_cards:
					self.game.end_turn()
					return self.create_board_state(), 0, self.is_game_over()
				else:
					return self.create_board_state(), self.calculate_reward()-100, self.is_game_over()
			elif action is 2:
				heropower = player.hero.power
				if heropower.is_usable():
					if heropower.requires_target():
						#targeting_agent.run(action, self.new_game)
						heropower.play(target=random.choice(heropower.targets))
						self.new_game = False
						self.check_remaining_actions()
						if learn_to_play_cards:
							return self.create_board_state(), 1, self.is_game_over()
						else:
							return self.create_board_state(), self.calculate_reward(), self.is_game_over()
					else:
						heropower.use()
						self.check_remaining_actions()
						if learn_to_play_cards:
							return self.create_board_state(), 1, self.is_game_over()
						else:
							return self.create_board_state(), self.calculate_reward(), self.is_game_over()
				if learn_to_play_cards:
					self.game.end_turn()
					return self.create_board_state(), 0, self.is_game_over()
				else:
					return self.create_board_state(), self.calculate_reward()-100, self.is_game_over()
			elif action is 3:
				# find the coin and play it
				for card in player.hand:
					if card.data.id == THE_COIN and card.is_playable():
						card.play()
						self.check_remaining_actions()
						if learn_to_play_cards:
							return self.create_board_state(), 1, self.is_game_over()
						else:
							return self.create_board_state(), self.calculate_reward(), self.is_game_over()
					if learn_to_play_cards:
						self.game.end_turn()
						return self.create_board_state(), 0, self.is_game_over()
					else:
						return self.create_board_state(), self.calculate_reward() - 100, self.is_game_over()
			else:
				to_play = self.draft_mage()[action-4]
				for card in player.hand:
					if card.id == to_play and card.is_playable():
						target = None
						if card.must_choose_one:
							card = random.choice(card.choose_cards)
						if card.requires_target():
							#targeting_agent.run(action, self.new_game)
							card.play(target=random.choice(card.targets))
							self.new_game = False
							if player.choice:
								choice = random.choice(player.choice.cards)
								print("Choosing card %r" % (choice))
								player.choice.choose(choice)
							self.check_remaining_actions()
							if learn_to_play_cards:
								return self.create_board_state(), 1, self.is_game_over()
							else:
								return self.create_board_state(), self.calculate_reward(), self.is_game_over()
						else:
							card.play(target=target)
							print("Playing %r on %r" % (card, target))
							if player.choice:
								choice = random.choice(player.choice.cards)
								print("Choosing card %r" % (choice))
								player.choice.choose(choice)
							self.check_remaining_actions()
							if learn_to_play_cards:
								return self.create_board_state(), 1, self.is_game_over()
							else:
								return self.create_board_state(), self.calculate_reward(), self.is_game_over()
				for minion in player.field:
					if minion.id == to_play and minion.can_attack():
						#targeting_agent.run(action, self.new_game)
						minion.attack(random.choice(minion.targets))
						self.new_game = False
						self.check_remaining_actions()
						if learn_to_play_cards:
							return self.create_board_state(), 1, self.is_game_over()
						else:
							return self.create_board_state(), self.calculate_reward(), self.is_game_over()
			if learn_to_play_cards:
				self.game.end_turn()
				return self.create_board_state(), 0, self.is_game_over()
			else:
				return self.create_board_state(), self.calculate_reward()-100, self.is_game_over()

		if self.game.ended:
			if learn_to_play_cards:
				self.game.end_turn()
				return self.create_board_state(), 0, self.is_game_over()
			else:
				return self.create_board_state(), self.calculate_reward(), self.is_game_over()
		#If there are no more actions pass the turn
		self.check_remaining_actions()
		if learn_to_play_cards:
			self.game.end_turn()
			return self.create_board_state(), 0, self.is_game_over()
		else:
			return self.create_board_state(), self.calculate_reward(), self.is_game_over()

	def calculate_reward(self):
		reward = 0
		reward += self.game.player2.hero.health
		reward += self.game.player2.hero.armor
		reward += self.game.player2.hero.atk
		if self.game.player2.weapon is not None:
			reward += self.game.player2.weapon.health
			reward += self.game.player2.weapon.atk
		reward -= self.game.player1.hero.health
		reward -= self.game.player1.hero.armor
		reward -= self.game.player1.hero.atk
		if self.game.player1.weapon is not None:
			reward -= self.game.player1.weapon.health
			reward -= self.game.player1.weapon.atk
		for minion in self.game.player2.field:
			reward += minion.health
			reward += minion.atk
		for minion in self.game.player1.field:
			reward -= minion.health
			reward -= minion.atk
		return reward

	def is_game_over(self):
		return self.game.ended

	def render(self):
		""" TODO: ADD RENDERING """
		self.renderer.render_board_state(game=self.game)

	def get_lives(self):
		return self.game.player1.hero.health

	def get_output_space(self):
		return 34


if __name__ == '__main__':
	env = HSEnv(True)

	while True:
		env.render()
		try:
			env.play_turn_random()
		except GameOver:
			env.reset()
		time.sleep(0.1)
