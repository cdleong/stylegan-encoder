import argparse
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pull the loss information out of saved output.')
    parser.add_argument('file_to_parse', help='file containing saved output')
    args = parser.parse_args()
    
    counters = []
    losses = []
    
    for line in open(args.file_to_parse, encoding='utf-8'):
        
        match1 = re.search('counter: (\d+)', line)
        match2 = re.search("Last Loss: (\d+.\d+)", line)
        if match1 and match2:
            counter = int(match1.group(1))
            loss = float(match2.group(1))                        
            counters.append(counter)
            losses.append(loss)
            
    encoder_data = pd.DataFrame(
            {
                "iteration": counters,
                 "loss":losses
             }
        )
    
    print(encoder_data.head(10))
    print(encoder_data.count())
    
    encoder_data.drop_duplicates(inplace=True)
    print("Dropping duplicates")
    print(encoder_data.head(10))
    print(encoder_data.count())
    
#     encoder_data = encoder_data.head(100)
#     print(encoder_data.count())

    save_folder = os.path.dirname(args.file_to_parse)
    print(f"saving images to {save_folder}")
    
    # whole curve
    plt.figure()    
    
    sns.lineplot(x="iteration",y="loss",data=encoder_data)
    
    plt.savefig(os.path.join(save_folder, "loss_curve.png"))
    
    # last 1k
    plt.figure()    
    
    sns.lineplot(x="iteration",y="loss",data=encoder_data.tail(1000))
    
    plt.savefig(os.path.join(save_folder,"loss_curve_last_1k.png"))
    
    
    # first 1k
    plt.figure()    
    
    sns.lineplot(x="iteration",y="loss",data=encoder_data.head(1000))
    
    plt.savefig(os.path.join(save_folder,"loss_curve_first_1k.png"))
    
    
    # last 100, mainly because Adam spikes so much
    plt.figure()    
    
    sns.lineplot(x="iteration",y="loss",data=encoder_data.tail(100))
    
    plt.savefig(os.path.join(save_folder,"loss_curve_last_100.png"))
    