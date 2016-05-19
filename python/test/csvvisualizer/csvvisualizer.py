from Tkinter import *
import csv
import sys

## Data
# todo to be moved in another file

if len(sys.argv) > 1:
    pathcsv = sys.argv[1]
else:
    pathcsv = '/Users/giulio/Dropbox (Personal)/Giulio/Education/4_PhD/Publications/publications.csv'

csvdata = []

with open(pathcsv, 'rb') as csvfile:
    rows = csv.reader(csvfile)
    csvdata = [row for row in rows]
numCols = max([len(row) for row in csvdata])

## Additional functions

def syncScroll(*args):
    for col in cols:
        col.yview_moveto(args[1])
        
def onSelectItem(event):
    widget = event.widget
    selection=widget.curselection()[0]
    value = widget.get(selection)
    print '\nLine ', selection
    for col in cols:
        col.config(exportselection=1)
    for col in cols:
        w = col.winfo_width()
        col.config(width = min(w, int((1.0 * currWidth)/numCols/9.5)))
    for col in cols:
        col.config(exportselection=0)
    for col in cols:
        value = col.get(selection)
        print '- %s' % value
        col.see(selection)
        col.selection_set(selection)

## Create  GUI

# main window
root = Tk()
root.title('CSV reader')
# root.geometry('800x600')
root.update()
currWidth = root.winfo_width()

scrollbar = Scrollbar(root)
scrollbar.pack( side = RIGHT, fill=Y )

pane = PanedWindow(sashrelief = RAISED)
pane.pack(fill=BOTH, expand=1)

cols = []
for col in range(0, numCols):
    currCol = Listbox(pane, width = 0, yscrollcommand = scrollbar.set, selectmode = BROWSE, borderwidth = 0)
    for line in range(0,len(csvdata)):
        if len(csvdata[line]) < col + 1:
            currCol.insert(END, '')
        else:
            currCol.insert(END, csvdata[line][col])
    currCol.pack(fill=BOTH, expand=1)
    currCol.bind("<ButtonRelease-1>", onSelectItem)
    pane.add(currCol)
    cols.append(currCol)
scrollbar.config( command = syncScroll )

# run the GUI
root.mainloop()