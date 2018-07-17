from LSTM import LSTMNetwork
from GRU import GRUNetwork
from LinearTransform import LinearTransform
import dynet as dy
from time import clock
import random
import matplotlib.pyplot as plt

class RNNTokenPredictor:
    def __init__(self,alphabet,num_layers=2,input_dim=3,hidden_dim=5,
                 reject_threshold = 0.01,RNNClass=LSTMNetwork,pc=None):

        self.alphabet = alphabet
        self.input_alphabet = list(self.alphabet) #there's no reason to have these two "different" things and it doesn't really matter except i've made a mess that needs cleaning up so i need both names atm
        self.end_token = "<EOS>"
#         self.begin_token = "<BOS>" #if decide to add BOS again, need to remember to apply it to every input sequence and treat state after <BOS> as first state
        self.internal_alphabet = self.input_alphabet + [self.end_token]#+[self.begin_token]
        self.int2char = list(self.internal_alphabet)
        self.char2int = {c:i for i,c in enumerate(self.int2char)}
        self.vocab_size = len(self.internal_alphabet)
        self.pc = pc if not None == pc else dy.ParameterCollection() 
        self.lookup = self.pc.add_lookup_parameters((self.vocab_size, input_dim))
        self.linear_transform = LinearTransform(hidden_dim,self.vocab_size,self.pc)
        self.rnn = RNNClass(num_layers=num_layers,input_dim=input_dim,hidden_dim=hidden_dim,pc=self.pc)
        self.reject_threshold = reject_threshold
        # not a real state because LSTM, GRUs both have
        # h values, which are in [-1,1]
        self.store_expressions() 
        # gets the initial state started, which is a roundabout way of enabling this:
        full_hidden_vec = self.rnn.initial_state.as_vec()
        self.sink_reject_vec = [2 for a in full_hidden_vec] 
        self.all_losses = []
        self.spec = {"alphabet":alphabet,"input_dim":input_dim,"num_layers":num_layers,
                     "hidden_dim":hidden_dim,"reject_threshold":reject_threshold,"RNNClass":RNNClass}

        #for converting vectors to states. self.slen = number of vectors in s, i.e. (num layers * (2 if lstm else 1))
    
    def param_collection(self):
        return self.pc

    @staticmethod
    def from_spec(spec,model): # model == parameter collection (i think)
        res = RNNTokenPredictor(spec["alphabet"],spec["num_layers"],spec["input_dim"],spec["hidden_dim"],spec["reject_threshold"],spec["RNNClass"],pc=model)
        return res

    def renew(self):
        dy.renew_cg()
        self.store_expressions()

    def store_expressions(self):
        self.rnn.store_expressions()
        self.linear_transform.store_expressions()
            
    def _char_to_input_vector(self,char):
        return self.lookup[self.char2int[char]]
            
    def _next_state(self,state,char):
        return self.rnn.next_state(state,self._char_to_input_vector(char))
    
    def _state_probability_distribution(self,state):
        return dy.softmax(self.linear_transform.apply(state.output())) 
    
    def _probability_of_char_after_state(self,state,char):
        return dy.pick(self._state_probability_distribution(state),self.char2int[char])
    
    def _state_accepts_next_char(self,state,next_char):
        return self._probability_of_char_after_state(state,next_char).value() > self.reject_threshold
    
    def _classify_state(self,state): #this one (and the next two) is/are just here if i ever want to extract
        return self._state_accepts_next_char(state,self.end_token)
    
    def get_first_RState(self): #if we're ever looking to extract on these
        return self.rnn.initial_state.as_vec(), self._classify_state(self.rnn.initial_state)
    
    def get_next_RState(self,vec,char): #also only here if we're ever looking to extract on these
        #verification, could get rid of
        if not char in self.input_alphabet:
            print("char for next vector not from input alphabet")
            return None     

        if vec == self.sink_reject_vec: #have already received input under prediction threshold
            return self.sink_reject_vec, False

        state = self.rnn.state_class(full_vec = vec, hidden_dim = self.rnn.hidden_dim)
        if not self._state_accepts_next_char(state,char): #have now received input under prediction threshold
            return self.sink_reject_vec, False

        state = self._next_state(state,char)
        return state.as_vec(), self._classify_state(state)
        
    def _word_is_over_input_alphabet(self,word):
        return next((False for c in word if not c in self.input_alphabet),True)
 
    def classify_word(self,word):
        #verification, could get rid of
        self.renew() # don't know if necessary
        if not self._word_is_over_input_alphabet(word):
            print("word is not over input alphabet")
            return False
        
        s = self.rnn.initial_state
        for c in word:
            if not self._state_accepts_next_char(s,c):
                return False
            s = self._next_state(s,c)
        return self._classify_state(s)

    def loss_on_word(self, word):
        s = self.rnn.initial_state
        loss = []
        for c in word:
            p = self._probability_of_char_after_state(s,c) #value between 0 and 1, ideally 1
            loss.append(-dy.log(p))
            
            s = self._next_state(s,c)
        p = self._probability_of_char_after_state(s,self.end_token)
        loss.append(-dy.log(p))
        loss = dy.esum(loss) 
        return loss
    
    def _train_one_word(self, trainer, word):
        self.renew()
        loss= self.loss_on_word(word)
        classification_loss_val = loss.value()
        loss.backward()
        trainer.update()
        return classification_loss_val
                
    def _train_group_individually(self,trainer,words):
        random.shuffle(words)
        loss_vals = []
        for w in words:
            l = self._train_one_word(trainer,w)
            loss_vals.append(l)
        return loss_vals
            
    def train_group(self,words,iterations,trainer=None,show=False,loss_every=100,print_time=True,learning_rate=0.001):
        if None == trainer:
            trainer = dy.AdamTrainer(self.pc)
            # print("learning rate:",trainer.learning_rate)
            # options: SimpleSGDTrainer, CyclicalSGDTrainer, 
            # MomentumSGDTrainer, AdagradTrainer, AdadeltaTrainer,
            # RMSPropTrainer, AdamTrainer, AmsgradTrainer, EGTrainer
            # there's also a generic class which i guess you can implement yourself if you want
        trainer.learning_rate = 0.001
        losses = []
        prev_avg_loss = 0
        start = clock()
        internal_start = start
        for i in range(iterations):
            new_losses = self._train_group_individually(trainer,words)
            losses += new_losses
            #manage learning rate
            avg_loss = sum(new_losses)/len(new_losses)
            ##curioz
            if i%loss_every == 1:
                print("classification loss:",avg_loss,end="")
                if print_time:
                    print("time:",clock()-internal_start)
                else:
                    print("")
                internal_start = clock()
            prev_avg_loss = avg_loss  
        self.all_losses += losses
        if print_time:
            print("total time:",clock()-start)
        if show:
            plt.scatter(range(len(losses)),losses)
            plt.title("classification losses for last set of epochs")
            plt.show()
            plt.scatter(range(len(self.all_losses)),self.all_losses)
            plt.title("classification losses since initiation")
            plt.show()
