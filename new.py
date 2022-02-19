
#import tensorflow.lite as tflite
import tflite_runtime.interpreter as tt
#from interpreter import Interpreter 
import numpy as np
#import IPython
#import pyaudio

# Test the model on random input data.
def fastspeech_infer(tflite_model_path, input_text):
  # Load the TFLite model and allocate tensors.
  interpreter = tt.Interpreter(model_path=tflite_model_path)
  print(interpreter)
  #print(interpreter.get_tensor_details())

  # Get input and output tensors.
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  a = {"pad": 0, "-": 1, "!": 2, "'": 3, "(": 4, ")": 5, ",": 6, ".": 7, ":": 8, ";": 9, "?": 10, " ": 11, "A": 12, "B": 13, "C": 14, "D": 15, "E": 16, "F": 17, "G": 18, "H": 19, "I": 20, "J": 21, "K": 22, "L": 23, "M": 24, "N": 25, "O": 26, "P": 27, "Q": 28, "R": 29, "S": 30, "T": 31, "U": 32, "V": 33, "W": 34, "X": 35, "Y": 36, "Z": 37, "a": 38, "b": 39, "c": 40, "d": 41, "e": 42, "f": 43, "g": 44, "h": 45, "i": 46, "j": 47, "k": 48, "l": 49, "m": 50, "n": 51, "o": 52, "p": 53, "q": 54, "r": 55, "s": 56, "t": 57, "u": 58, "v": 59, "w": 60, "x": 61, "y": 62, "z": 63, "@AA": 64, "@AA0": 65, "@AA1": 66, "@AA2": 67, "@AE": 68, "@AE0": 69, "@AE1": 70, "@AE2": 71, "@AH": 72, "@AH0": 73, "@AH1": 74, "@AH2": 75, "@AO": 76, "@AO0": 77, "@AO1": 78, "@AO2": 79, "@AW": 80, "@AW0": 81, "@AW1": 82, "@AW2": 83, "@AY": 84, "@AY0": 85, "@AY1": 86, "@AY2": 87, "@B": 88, "@CH": 89, "@D": 90, "@DH": 91, "@EH": 92, "@EH0": 93, "@EH1": 94, "@EH2": 95, "@ER": 96, "@ER0": 97, "@ER1": 98, "@ER2": 99, "@EY": 100, "@EY0": 101, "@EY1": 102, "@EY2": 103, "@F": 104, "@G": 105, "@HH": 106, "@IH": 107, "@IH0": 108, "@IH1": 109, "@IH2": 110, "@IY": 111, "@IY0": 112, "@IY1": 113, "@IY2": 114, "@JH": 115, "@K": 116, "@L": 117, "@M": 118, "@N": 119, "@NG": 120, "@OW": 121, "@OW0": 122, "@OW1": 123, "@OW2": 124, "@OY": 125, "@OY0": 126, "@OY1": 127, "@OY2": 128, "@P": 129, "@R": 130, "@S": 131, "@SH": 132, "@T": 133, "@TH": 134, "@UH": 135, "@UH0": 136, "@UH1": 137, "@UH2": 138, "@UW": 139, "@UW0": 140, "@UW1": 141, "@UW2": 142, "@V": 143, "@W": 144, "@Y": 145, "@Z": 146, "@ZH": 147, "eos": 148}
  l=[]
  temp=''
  for i in input_text.lower():
    if i== ' ' :
      if temp!=' ':
        l.append(a[i])
    else:
      l.append(a[i])
    temp=i

  l.append(148) 
  

  interpreter.resize_tensor_input(input_details[0]['index'],[1, len(l)])
  interpreter.resize_tensor_input(input_details[1]['index'],[1])
  interpreter.resize_tensor_input(input_details[2]['index'],[1])
  interpreter.resize_tensor_input(input_details[3]['index'], [1])
  interpreter.resize_tensor_input(input_details[4]['index'], [1])
  interpreter.allocate_tensors()
  #input_data = fastspeech_prepare_input(l)
  
  #input_data = fastspeech_prepare_input(input_ids)

  x=np.array(l, dtype=np.int32)
  x.resize(1,len(x))
  y=np.array([0],dtype=np.int32)
  z=np.array([1.0],dtype=np.float32)
  w=np.array([1.0],dtype=np.float32)
  v=np.array([1.0],dtype=np.float32)

  interpreter.set_tensor(0, x.astype(np.int32))
  interpreter.set_tensor(1, y)
  interpreter.set_tensor(2, z)
  interpreter.set_tensor(3, w)
  interpreter.set_tensor(4,v)




  interpreter.invoke()

  # The function `get_tensor()` returns a copy of the tensor data.
  # Use `tensor()` in order to get a pointer to the tensor.
  return (interpreter.get_tensor(output_details[0]['index']),
          interpreter.get_tensor(output_details[1]['index']))
  
def run_melgan(mel_spec, quantization):
    model_name = f'mb_melgan.tflite'
    
    feats = np.expand_dims(mel_spec, 0)
    interpreter = tt.Interpreter(model_path=model_name)

    input_details = interpreter.get_input_details()

    output_details = interpreter.get_output_details()

    print(input_details,'\n\n', output_details)

    interpreter.resize_tensor_input(input_details[0]['index'],  [1, feats.shape[1], feats.shape[2]], strict=True)
    interpreter.allocate_tensors()

    interpreter.set_tensor(input_details[0]['index'], feats)

    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    
    return output

text =  "Hello  how are you "    
_, tac_output = fastspeech_infer('fastspeech2_quan.tflite', text)
tac_output = np.squeeze(tac_output)
sample_rate = 22050

waveform = run_melgan(tac_output, '')
print(waveform, type(waveform))
waveform = np.squeeze(waveform)


#IPython.display.display(IPython.display.Audio(waveform, rate=sample_rate))