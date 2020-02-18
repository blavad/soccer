"""
Base class of soccer games.
"""
import os
import cv2
import math
import numpy as np

import gym
from gym import spaces, logger
from gym.utils import seeding

import soccer
from soccer.core import Team1, Team2
 
class BaseSoccerEnv(gym.Env):
 
    def __init__(self, width=300, height=200, height_goal=None, nb_pl_team1=1, nb_pl_team2=1, type_config="discrete"):
        
        self.score = np.array([0,0])
        
        # Field parameters
        self.width = width
        self.height = height
        self.w_field = width
        self.h_field = height
        self.h_goal = self.h_field//2 if height_goal is None else height_goal
        self.goal_pos = (self.h_field//2 - self.h_goal//2, self.h_field//2 + (self.h_goal-self.h_goal//2))
        self.type_config = type_config
        
        # Players parameters
        self.size_player = min(width//5, height//5)
        self.size_player_w = int(self.size_player*0.36)
        self.team = [Team1(nb_pl_team1).init_config(self.w_field, self.h_field, size_pl=self.size_player, type_config=self.type_config), Team2(nb_pl_team2).init_config(self.w_field, self.h_field, size_pl=self.size_player, type_config=self.type_config)]
        self.all_players[np.random.randint(self.n_players)].has_ball=True
        self.update_field()
        self.size_ball =  self.size_player//3
    
        self.done_flag = False
    
        self.init_assets()
            
        self.viewer = None
        
    @property
    def team1(self):
        return self.team[0]
    
    @property
    def team2(self):
        return self.team[1]
    
    @property
    def n_players(self):
        return len(self.team1) + len(self.team2)
    
    @property
    def all_players(self):
        return self.team1.player + self.team2.player
 
 
    def reset(self):
        self.team[0] = self.team[0].init_config(self.w_field, self.h_field, size_pl=self.size_player, type_config=self.type_config)
        self.team[1] = self.team[1].init_config(self.w_field, self.h_field, size_pl=self.size_player, type_config=self.type_config)
        self.all_players[np.random.randint(self.n_players)].has_ball=True
        self.done_flag = False
        self.update_field()
        # self.viewer = None
        
        return self.state

    def step(self, actions):
        action = []
        try :
            actions = list(actions)
        except TypeError :
            actions = [actions]          
        for act in actions:
            assert self.action_space.contains(act), "%r (%s) invalid" % (act, type(act))
            action += [self.__class__.actions[act]]
        
        self.update_state(action)
        rew, done = self.reward(action)
        
        self.update_field()
        
        return self.state, rew, done, {}

 
    ########## RENDER PART ##############
    def render(self, mode='human'):
        if mode == 'human':
            return self.render_human(mode)
        elif mode == 'rbg_array':
            return self.render_rgb_array()
        return self.render_array()

    def render_human(self, mode='human'):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()
        return self.viewer.imshow(self.render(mode='rbg_array')[:,:,:3])

    def render_array(self):
        print(self.field)

    def render_rgb_array(self):    
        color = np.array([50,150,50])
        return np.concatenate((self.renderGame(color_background=color), self.renderInfos(score=self.score, color_background=color-50)), axis=0)
        

    def renderGame(self, color_background=[50,200,50]):
        img = np.full(
            (self.height, self.width, 4),
            255,
            dtype=np.uint8,
        )
        img[:,:,:3] = color_background
        for p in self.all_players:
            img = self.draw_player(img, p)
        img = self.draw_ball(img)
        img = self.draw_goal(img)
        return img
        
    def renderInfos(self, score=None, color_background=[50,200,200]):
        height = self.width//6
        infosImg = np.full(
            (height, self.width, 4),
            255,
            dtype=np.uint8,
        )
        infosImg[:,:,:3] = color_background
        return self.displayInfos(infosImg, score)
         

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
            
    def displayInfos(self, img,  score):
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0,0,0)
        cv2.putText(img, "Blue {} - {} Red".format(self.score[0],self.score[1]), (2*self.width//7, self.width//10), font, min(1., 0.2*self.w_field), color, 1, cv2.LINE_AA)
        return img