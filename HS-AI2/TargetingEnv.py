from HSEnv import HSEnv
import random

class TargetEnv:
	def __init__(self, hsenv):
		self.hs_game = hsenv

	def step(self, action, target):
		player = self.hs_game.game.current_player
		targetEntity = None
		if target == 0:
			targetEntity = player.hero
		elif target == 1:
			targetEntity = player.opponent.hero
		else:
			if target < 9:
				if len(player.field) > target-2:
					targetEntity = player.field[target-2]
				else:
					self.hs_game.game.end_turn()
					return 0#self.hs_game.calculate_reward()-100
			else:
				if len(player.opponent.field) > target-9:
					targetEntity = player.opponent.field[target-9]
				else:
					self.hs_game.game.end_turn()
					return 0#self.hs_game.calculate_reward()-100
		if targetEntity is None:
			self.hs_game.game.end_turn()
			return 0#self.hs_game.calculate_reward()-100
		if action is 1:
			if player.hero.can_attack(targetEntity):
				player.hero.attack(target=targetEntity)
			else:
				self.hs_game.game.end_turn()
				return 0#self.hs_game.calculate_reward()-100
		elif action is 2:
			if targetEntity in player.hero.power.targets:
				player.hero.power.use(target=targetEntity)
			else:
				self.hs_game.game.end_turn()
				return 0#self.hs_game.calculate_reward()-100
		else:
			to_play = self.hs_game.draft_mage()[action - 4]
			for card in player.hand:
				if card.id == to_play and card.is_playable():
					if targetEntity in card.targets:
						card.play(target=targetEntity)
					else:
						self.hs_game.game.end_turn()
						return 0#self.hs_game.calculate_reward()-100
					break
			for minion in player.opponent.field:
				if minion.id == to_play and minion.can_attack():
					if minion.can_attack(target=targetEntity):
						minion.attack(target=targetEntity)
					else:
						self.hs_game.game.end_turn()
						return 0#self.hs_game.calculate_reward()-100
		return 10#self.hs_game.calculate_reward()

	def create_board_state(self, action):
		state = []
		state.append(action)
		for i in self.hs_game.create_board_state():
			state.append(i)
		return state

	def get_lives(self):
		return self.hs_game.get_lives()

	def get_output_space(self):
		return 16

	def get_action_meanings(self):
		strings = []

		strings.append("Friendly Hero")
		strings.append("Opponent Hero")
		for i in range(7):
			strings.append("Friendly Minion" + str(i))
		for i in range(7):
			strings.append("Opponent Minion" + str(i))

		return strings

	def render(self):
		return
