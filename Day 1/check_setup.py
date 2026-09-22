"""Step 1: run this first. It checks the environment and the model connection.""" 
import sys 
from config import client, MODEL, PROVIDER 
  
print("Python version :", sys.version.split()[0]) 
print("Provider       :", PROVIDER) 
print("Model          :", MODEL) 
print("Calling the model ...") 
  
response = client.chat.completions.create( 
    model=MODEL, 
    messages=[{"role": "user", "content": "Reply with exactly these two words: SETUP"}],
    temperature=0, 
) 
print("Model replied  :", response.choices[0].message.content.strip()) 
print("\nSetup check finished.")