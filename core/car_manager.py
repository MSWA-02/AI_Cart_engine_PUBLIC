import os

import pygame
from my_sprites.AI_car import AI_car
from constants import WHITE, SPEED, MAX_SPEED, TILESIZE, TURN_SPEED, BREAK_SPEED, FRICTION_SPEED,PATH_SAVES
import copy
from pathlib import Path
import numpy as np

SPAWN_Y_JITTER_PX = 12

class Car_manager():
    def __init__(self, pocet_aut, pocet_generaci, max_time, cars_to_next,save_as, load_from):
        self.pocet_aut = pocet_aut
        self.pocet_generaci = pocet_generaci
        self.max_time = max_time
        self.cars_to_next = cars_to_next
        self.save_as = save_as
        self.load_from = load_from

        self.epoch = 0
        self.cur_epoch = 0
        self.total_time = 0
        self.running = False # tykas se zda bezi nejaká epocha
        self.pustene = False # týká se zda bezí celkvoe trening

        self.sprite_list = list() # registrujeme do vsech - abych je pak byl schopen prebrat
        self.sprite_running = pygame.sprite.Group() # praucjeme jen s running ve kole
        self.best_cars_list = list() # bezst brain z minulé epochy

    def setup(self, pocet_aut, pocet_generaci, max_time, cars_to_next, save_as, load_from):
        self.pocet_aut = pocet_aut
        self.pocet_generaci = pocet_generaci
        self.max_time = max_time
        self.cars_to_next = cars_to_next
        self.save_as = save_as
        self.load_from = load_from

    def setup_next_epoch(self):
        self.score_cars()# oskoruji auta
        self.best_cars_list = [c for c in self.best_cars_list[0:self.cars_to_next]]# ulozim si strnaou nejlepsích n

        self.sprite_list = list()# vyresetuji cekový list
        for c in self.sprite_running:# vymazu vse
            c.kill()
        self.reset_brains()
        self.total_time = 0
        self.cur_epoch += 1

        base_x = TILESIZE * 4 + int(TILESIZE / 2)
        base_y = TILESIZE * 8 + int(TILESIZE / 2)

        for i in range(self.cars_to_next):
            y = base_y + int(np.random.randint(-SPAWN_Y_JITTER_PX, SPAWN_Y_JITTER_PX + 1))
            c = AI_car(base_x, y, 10, 20,
                       copy.deepcopy(self.best_cars_list[i].brain),  180+np.random.randint(-45,+45))
            self.sprite_list.append(c)
            self.sprite_running.add(c)

        # doplneneí zmutovanou generací prvních n aut:
        for j in range(self.pocet_aut-self.cars_to_next):
            i = j % self.cars_to_next
            c = AI_car(TILESIZE * 4 + int(TILESIZE / 2), TILESIZE * 8 + int(TILESIZE / 2), 10, 20,
                       copy.deepcopy(self.best_cars_list[i].brain))

            c.brain.mutate()
            self.sprite_list.append(c)
            self.sprite_running.add(c)

        self.running = True

    def score_cars(self):
        self.best_cars_list = list()
        self.best_cars_list = sorted(self.sprite_list, key=lambda obj: obj.brain.score, reverse=True)

    def add_defaultbrain(self, brain):
        self.defaultbrain = brain
        self.reset_brains()

    # start - od začátku vše, zresetuje co jde a nastaví auta znova
    def start(self):
        # zapnu start a vymazu vsechny informace pokud nekde jsou:
        if len(self.sprite_list) > 0:
            self.sprite_list = list()  # registrujeme do vsech - abych je pak byl schopen prebrat
        if len(self.sprite_running) > 0:
            for c in self.sprite_running:
                c.kill()
            self.sprite_running = pygame.sprite.Group()  # praucjeme jen s running ve kole

        self.cur_epoch = 0
        self.total_time = 0

        base_x = TILESIZE * 4 + int(TILESIZE / 2)
        base_y = TILESIZE * 8 + int(TILESIZE / 2)

        # a vytvořím auta:
        for i in range(self.pocet_aut):
            y = base_y + int(np.random.randint(-SPAWN_Y_JITTER_PX, SPAWN_Y_JITTER_PX + 1))
            c = AI_car(base_x, y, 10, 20, self.brain_list[i], 180+np.random.randint(-45,+45))
            self.sprite_list.append(c)
            self.sprite_running.add(c)

        # a zapnu tréning
        self.running = True
        self.pustene = True

    def get_sprite_group(self):
        return self.sprite_running

    def reset_brains(self):
        self.brain_list = [self.defaultbrain() for _ in range(self.pocet_aut)]

    def draw(self, screen):
        self.sprite_running.draw(screen)

    def stop(self):
        self.running = False

    def autosave(self):
        self.save(self.save_as)

    def save(self, file):
        self.score_cars()
        print(f"probiha save souboru {file}")
        print(os.getcwd())
        print(self.best_cars_list[0].brain.get_parameters())# nejlepsí je na prvním íste :)
        np.savez(Path(PATH_SAVES+file), **self.best_cars_list[0].brain.get_parameters())

    def load(self, file = "userbrain.npz"):
        self.running = False
        print(f"probiha load souboru {file}")
        params =  np.load(Path(PATH_SAVES+file))
        print({key: params[key] for key in params.files})
        self.reset_brains()

        if len(self.sprite_list) > 0:
            self.sprite_list = list()  # registrujeme do vsech - abych je pak byl schopen prebrat
        if len(self.sprite_running) > 0:
            for c in self.sprite_running:
                c.kill()
            self.sprite_running = pygame.sprite.Group()  # praucjeme jen s running ve kole

        self.cur_epoch = 0
        self.total_time = 0
        base_x = TILESIZE * 4 + int(TILESIZE / 2)
        base_y = TILESIZE * 8 + int(TILESIZE / 2)

        # a vytvořím auta:
        for i in range(self.pocet_aut):
            y = base_y + int(np.random.randint(-SPAWN_Y_JITTER_PX, SPAWN_Y_JITTER_PX + 1))
            c = AI_car(base_x, y, 10, 20, self.brain_list[i],  180+np.random.randint(-45,+45))
            c.brain.set_parameters(params)
            if i>0:
                c.brain.mutate()
            self.sprite_list.append(c)
            self.sprite_running.add(c)

        # a zapnu tréning
        self.running = True
        self.pustene = True

    def update(self,  dt, keys, blocks):
        if self.running:
            for c in self.sprite_running:
                c.update(dt, keys, blocks)
                # nárazy do zdí
                hit = pygame.sprite.spritecollideany(c, blocks.sprites)
                if hit is not None:
                    c.running = False

            self.total_time += dt


        # enco jako if total time> max time - new epoch
        # ted musím doresit epochy, kolize a save a load
        if self.total_time > self.max_time  and self.pustene:
            self.running = False # timer pro epochu

            if self.cur_epoch < self.pocet_generaci:
                print(f"-----epocha {self.cur_epoch}------------------------")
                #for sprite in self.sprite_list:
                #    print(sprite.brain.parameters)
                #print(f"----------------------------------------------------")
                self.setup_next_epoch()
            else:
                self.pustene = False
                self.autosave()


        #print(f"total time: {self.total_time}")
        #print(f"setting, pocet aut: {self.pocet_aut}, pocet generaci: {self.pocet_generaci}, epoch: {self.epoch}")
        #print(f"vnitrni data: len sprite list: {len(self.sprite_list)}, len sprite running: {len(self.sprite_running)}")
