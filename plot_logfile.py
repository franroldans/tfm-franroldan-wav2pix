import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


parser = argparse.ArgumentParser()
parser.add_argument("--save_path", default='')
parser.add_argument("--type", default='gan')
args = parser.parse_args()

output_filename = os.path.normpath(os.path.join('logs', args.save_path))
d_loss = []
g_loss = []
real_loss = []
fake_loss = []
D = []
DofG = []
with open(output_filename + '/logFile.log', 'rb') as logfile:
    for line in logfile:
        items = line.split(', ')
        if args.type == 'wgan':
            d_loss.append(float(items[2].split('= ')[1]))
            g_loss.append(float(items[3].split('= ')[1]))
            real_loss.append(float(items[4].split('= ')[1]))
            fake_loss.append(float(items[5].split('= ')[1].replace('\n', '')))
        elif args.type == 'gan':
            d_loss.append(float(items[1].split('= ')[1]))
            g_loss.append(float(items[2].split('= ')[1]))
            D.append(float(items[3].split('= ')[1]))
            DofG.append(float(items[4].split('= ')[1].replace('\n', '')))

plt.clf()
plt.title('Generator vs Discriminator Loss')
plt.xlabel('Mini-batches')
plt.ylabel('Loss')
plt.ylim([0, max(max(d_loss),max(g_loss))])
plt.xlim([0, len(d_loss)])
plt.plot(d_loss, 'r-')
plt.plot(g_loss, 'b-')
red_patch = mpatches.Patch(color='red', label='Disc Loss')
blue_patch = mpatches.Patch(color='blue', label='Gen loss')
plt.legend(handles=[red_patch, blue_patch])
plt.savefig('losses_plot.png')