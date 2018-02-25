# CAN BE RUN ONLY ON CODESKULPTOR.ORG

import simplegui
import random

# load card sprite - 949x392 - source: jfitz.com
CARD_SIZE = (73, 98)
CARD_CENTER = (36.5, 49)
card_images = simplegui.load_image("http://commondatastorage.googleapis.com/codeskulptor-assets/cards.jfitz.png")

CARD_BACK_SIZE = (71, 96)
CARD_BACK_CENTER = (35.5, 48)
card_back = simplegui.load_image("http://commondatastorage.googleapis.com/codeskulptor-assets/card_back.png")    

# initialize some useful global variables
in_play = False
outcome = ""
score = 0
ins = ""

# define globals for cards
SUITS = ('C', 'S', 'H', 'D')
RANKS = ('A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K')
VALUES = {'A':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, 'T':10, 'J':10, 'Q':10, 'K':10}


# define card class
class Card:
    def __init__(self, suit, rank):
        if (suit in SUITS) and (rank in RANKS):
            self.suit = suit
            self.rank = rank
        else:
            self.suit = None
            self.rank = None
            print "Invalid card: ", suit, rank

    def __str__(self):
        return self.suit + self.rank

    def get_suit(self):
        return self.suit

    def get_rank(self):
        return self.rank

    def draw(self, canvas, pos):
        card_loc = (CARD_CENTER[0] + CARD_SIZE[0] * RANKS.index(self.rank), 
                    CARD_CENTER[1] + CARD_SIZE[1] * SUITS.index(self.suit))
        canvas.draw_image(card_images, card_loc, CARD_SIZE, [pos[0] + CARD_CENTER[0], pos[1] + CARD_CENTER[1]], CARD_SIZE)
        
# define hand class
class Hand:
    def __init__(self):
        self.value = 0	# create Hand object
        self.hand = []

    def __str__(self):
        
        s =""
        for card in self.hand:
            s += str(card.get_suit()) + str(card.get_rank()) + " "
        return "Hand contains " + s
    
    def add_card(self, card):
        self.hand.append(card)	# add a card object to a hand

    def get_value(self):
        # count aces as 1, if the hand has an ace, then add 10 to hand value if it doesn't bust
        # compute the value of the hand, see Blackjack video
        
        self.value = 0
        i=0
        for card in self.hand:
            self.value += VALUES[card.get_rank()]
            if card.get_rank() == "A":
                i=1
            
        if i == 1:
            if (self.value + 10) <= 21:
                self.value += 10
                
        return self.value
    
    def newhand(self):
        rem = []
        for card in self.hand:
            rem.append(card)
        for card in rem:
            self.hand.remove(card)    
            
            
            
    def draw(self, canvas, pos):
        
        for card in self.hand:
            card.draw(canvas,pos)
            pos[0] += 100
            
            # draw a hand on the canvas, use the draw method for cards
 
       
 
        
# define deck class 
class Deck:
    def __init__(self):
        self.deck = []	# create a Deck object
        for i in SUITS:
            for j in RANKS:
                self.deck.append(Card(i , j))

    def shuffle(self):
        # shuffle the deck 
            # use random.shuffle()
        random.shuffle(self.deck)

    def deal_card(self):
        return self.deck.pop()	# deal a card object from the deck
    
    def __str__(self):
            # return a string representing the deck
        s = ""
        for card in deck:
            s = s + str(card.get_suit()) + str(card.get_rank()) 
        return "Deck contains " + s           
        

dealer = Hand()
player = Hand()
deck = Deck()

#define event handlers for buttons
def deal():
    global outcome, in_play, dealer, player , deck,ins, score
    if in_play:
        score -=1
    
    
    dealer.newhand()
    player.newhand()
    #deck = Deck()
    deck.shuffle()
    dealer.add_card(deck.deal_card())
    player.add_card(deck.deal_card())
    player.add_card(deck.deal_card())
    ins = "Hit or Stand?"
    
    # your code goes here
    
    in_play = True

def hit():
        # replace with your code below
    global outcome, ins, dealer, player , deck,score, in_play
    if in_play:
        
        player.add_card(deck.deal_card())
        
        dealer.add_card(deck.deal_card())
        
        if player.get_value() <= 21:
    
            if player.get_value() > dealer.get_value() :
                outcome = "You win"
                score += 1
                ins = "New Deal?"
            else:
                outcome = "You lose"
                score -= 1
                ins = "New Deal?"
        else:
            outcome = "You have busted"
            score -= 1
            ins = "New Deal?"
        in_play = False                
                
 
    # if the hand is in play, hit the player
   
    # if busted, assign a message to outcome, update in_play and score
       
def stand():
    global outcome, ins, dealer, player , deck, score, in_play
    if in_play:
        while dealer.get_value() <= 17:
            dealer.add_card(deck.deal_card())
            
        if dealer.get_value() > 21:
            outcome = "Dealer is busted, You win"
            score += 1
            ins = "New Deal?"
            in_play = False
        else:    
            if player.get_value() > dealer.get_value():
                outcome = "You win"
                score += 1
                ins = "New Deal?"
            else:
                outcome = "You lose"
                score -= 1
                ins = "New Deal?"
    # if hand is in play, repeatedly hit dealer until his hand has value 17 or more
        in_play = False
    # assign a message to outcome, update in_play and score

# draw handler    
def draw(canvas):
    # test to make sure that card.draw works, replace with your code below
    global outcome, ins, dealer, player
    canvas.draw_image(card_back, [35.5, 48], [71 , 96], [100 , 200], [71 , 96])
#    card = Card("S", "A")
 #   card.draw(canvas, [300, 300])
    canvas.draw_text("Blackjack", [100,50], 35, "Red")
    canvas.draw_text("Score = "+str(score), [350,80], 24, "Black")
    canvas.draw_text(outcome, [300,120], 24, "Black")
    canvas.draw_text("Dealer", [100,120], 24, "Black")
    canvas.draw_text(ins, [300,350], 24, "Black")
    canvas.draw_text("Player", [100,350], 24, "Black")
    
    dealer.draw(canvas, [200, 200])
    player.draw(canvas, [200, 400])
    
    


# initialization frame
frame = simplegui.create_frame("Blackjack", 600, 600)
frame.set_canvas_background("Green")

#create buttons and canvas callback
frame.add_button("Deal", deal, 200)
frame.add_button("Hit",  hit, 200)
frame.add_button("Stand", stand, 200)
frame.set_draw_handler(draw)


# get things rolling
deal()
frame.start()


# remember to review the gradic rubric
