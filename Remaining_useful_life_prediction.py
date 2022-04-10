def convert(df):
    df = df[[f for f in range(0,26)]]
    df.columns = ['ID', 'Cycle', 'OpSet1', 'OpSet2', 'OpSet3',
                        'SensorMeasure1', 'SensorMeasure2', 'SensorMeasure3',
                        'SensorMeasure4', 'SensorMeasure5', 'SensorMeasure6',
                        'SensorMeasure7', 'SensorMeasure8', 'SensorMeasure9',
                        'SensorMeasure10', 'SensorMeasure11', 'SensorMeasure12',
                        'SensorMeasure13', 'SensorMeasure14', 'SensorMeasure15',
                        'SensorMeasure16', 'SensorMeasure17', 'SensorMeasure18',
                        'SensorMeasure19', 'SensorMeasure20', 'SensorMeasure21']
    return df

def generateY(df,n):
    max_cycles_df = df.groupby(['ID'], sort=False)['Cycle'].max().reset_index().rename(columns={'Cycle':'MaxCycleID'})
    # max_cycles_df.head()
    FD001_df = pd.merge(df, max_cycles_df, how='inner', on='ID')
    FD001_df['RUL'] = FD001_df['MaxCycleID'] - FD001_df['Cycle']
    out=pd.DataFrame()
    out['RUL']=FD001_df['RUL'].apply(lambda x: 0 if x-n>0 else 1)
    
    return [out,max_cycles_df]
    
def sensorplt(df,df2):
    d=df['ID'].unique()[0]
    # df=X[X['ID']==X['ID'].unique()[0]]
    import matplotlib.pyplot as plt
    df.sort_values(by=['Cycle'],inplace=True)
    df.drop('ID',axis=1,inplace=True)
    df.set_index(['Cycle'],inplace=True)
    df2.sort_values(by=['Cycle'],inplace=True)
    df2.drop('ID',axis=1,inplace=True)
    df2.set_index(['Cycle'],inplace=True)
    col=df.columns
    ax=[[],]*len(col)
    fig, axs = plt.subplots(nrows=4,ncols=6,constrained_layout=True)
    i=0
    for ax in axs.flat:
        ax.plot(list(df.index),df[col[i]],'b',label='Actual')
        ax.plot(list(df2.index),df2[col[i]],'r',linewidth=2,label='Smooth')
        ax.set_title(col[i], fontsize=10)
        i=i+1
        ax.legend()
    fig.canvas.manager.full_screen_toggle()
    plt.savefig(str(d)+'.png')
    plt.close(fig)
    # print(df)
    
def smoothie(df,n):
    from scipy.signal import savgol_filter

    X_modi=pd.DataFrame(columns=df.columns)

    for iden in df['ID'].unique():
        x_dum=df[df['ID']==iden].sort_values(by=['Cycle'])
        for col in columns:
            x_dum[col]=savgol_filter(x_dum[col], n, 3)
        X_modi=X_modi.merge(x_dum,how='outer')
    return X_modi

def plot3d(X_set, y_set,name):
    from matplotlib.colors import ListedColormap
    fig = plt.figure(figsize=(18,9))
    
    ax = plt.axes(projection ='3d')

    for i, j in enumerate(np.unique(y_set)):
        data=X_set[(y_set['RUL'] == j),:]
        z = data[:,0]
        x = data[:,1]
        y = data[:,2]         
        ax.scatter(x,y,z,c = ListedColormap(('blue', 'red'))(i), label = j)
    # fig.subplots_adjust(top=1.2, bottom=-.2) 
    ax.set_xlabel('PC1') # for Xlabel
    ax.set_ylabel('PC2') 
    ax.set_zlabel('PC3') 
    ax.legend(['Safe','Critical'])
     
    # syntax for plotting
    ax.set_title(name.upper(),fontdict={'fontsize': 16, 'fontweight': 'bold'})
    plt.savefig(name+'.png')
    plt.close(fig)
    
def plot3dcomp(X_set, y_set, y_pred , name):
    from matplotlib.colors import ListedColormap
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    corct=y_set.reset_index().drop('index',axis=1).join(y_pred,how='outer',rsuffix='_pred')
    corct['comp']=corct.apply(lambda x: x[0]+x[1],axis=1)
    
    for i, j in enumerate(np.unique(y_set)):
        data=X_set[(y_set['RUL'] == j),:]
        z = data[:,0]
        x = data[:,1]
        y = data[:,2]         
        ax1.scatter(x,y,z,c = ListedColormap(('green', 'red'))(i), label = j)
    for i, j in enumerate(np.unique(corct['comp'])):
        data=X_set[(corct['comp'] == j),:]
        z = data[:,0]
        x = data[:,1]
        y = data[:,2]         
        ax2.scatter(x,y,z,c = ListedColormap(('green','blue','red' ))(i), label = j)
    # fig.subplots_adjust(top=1.2, bottom=-.2) 
    ax1.set_xlabel('PC1') # for Xlabel
    ax1.set_ylabel('PC2') 
    ax1.set_zlabel('PC3') 
    ax1.legend(['Critical','Safe'])
    ax1.set_title('Test',fontdict={'fontsize': 16, 'fontweight': 'bold'})
    
    ax2.set_xlabel('PC1') # for Xlabel
    ax2.set_ylabel('PC2') 
    ax2.set_zlabel('PC3') 
    ax2.legend(['Safe-correct','Incorrect','Critical-correct'])
    ax2.set_title('Prediction',fontdict={'fontsize': 16, 'fontweight': 'bold'})
     
    # syntax for plotting
    fig.suptitle(name.upper(),fontsize=16,fontweight='bold')
    fig.canvas.manager.full_screen_toggle()
    plt.savefig(name+'.png')
    plt.close(fig)
    
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from sklearn.model_selection import train_test_split

testpath=r'test_FD001.txt'
trainpath=r'train_FD001.txt'
testrul=r'RUL_FD001.txt'
warntime=20

columns = ['OpSet1', 'OpSet2', 'SensorMeasure2', 'SensorMeasure3','SensorMeasure4', 'SensorMeasure7', 
            'SensorMeasure8', 'SensorMeasure9', 'SensorMeasure11', 'SensorMeasure12', 'SensorMeasure13',
            'SensorMeasure14', 'SensorMeasure15', 'SensorMeasure17', 'SensorMeasure20', 'SensorMeasure21']

X_1 = convert(pd.read_csv(trainpath, sep=' ', header = None))
[Y_1,a]=generateY(X_1,warntime)
X_modi=smoothie(X_1,51)

X_2=convert(pd.read_csv(testpath, sep=' ', header = None))
X_modi2=smoothie(X_2,21)
X_test_modi=X_modi2[columns]

[a,b]=generateY(X_2,warntime)
Y_2=pd.read_csv(testrul, sep=' ', header = None)
Y_2.reset_index(inplace=True)
Y_2.columns=['ID','MaxCycleID','d']
Y_2.drop('d',axis=1,inplace=True)
Y_2['ID']=Y_2['ID']+1
Y_2['MaxCycleID']=b['MaxCycleID']+Y_2['MaxCycleID']
FD001_df = pd.merge(X_2, Y_2, how='inner', on='ID')
FD001_df['RUL'] = FD001_df['MaxCycleID'] - FD001_df['Cycle']
Y_test=pd.DataFrame()
Y_test['RUL']=FD001_df['RUL'].apply(lambda x: 0 if x-warntime>0 else 1)

for iden in X_1['ID'].unique():
    sensorplt(X_1[X_1['ID']==iden],X_modi[X_modi['ID']==iden])

X_train_modi=X_modi[columns]
Y_train=Y_1

#%%

from sklearn.linear_model import LinearRegression
model=LinearRegression()

reg=model.fit(X_train_modi,Y_train)

coeff=model.coef_

#%%
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
 
X_train = sc.fit_transform(X_train_modi)
X_test = sc.transform(X_test_modi)

from sklearn.decomposition import PCA
 
# pca = PCA(.9)
pca = PCA(n_components =3)
 
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_

#%% regression classification

from sklearn.linear_model import LogisticRegression 
 
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
 
cm = confusion_matrix(Y_test, Y_pred)
accuracy=classifier.score(X_test, Y_test)


#%% Plot 3d if PCA uses only 3 components

# plot3d(X_test, Y_test,'Test')
plot3d(X_train, Y_train,'Train')
pred=pd.DataFrame(data=Y_pred,columns=['RUL',])

plot3dcomp(X_test, Y_test, pred , 'Test Vs Prediction')

#%%
from matplotlib.colors import ListedColormap
data=X_2.join(pred,how='outer')
df=data[['ID','Cycle','RUL']]
fig, axs = plt.subplots(nrows=4,ncols=3,constrained_layout=True)
iden = data['ID'].unique()[31:43]
k=0
for ax in axs.flat:
    x_dum=df[df['ID']==iden[k]].sort_values(by=['Cycle'])
    for i, j in enumerate(np.unique(data['RUL'])):
        ax.scatter(x_dum[x_dum['RUL']==j]['Cycle'],x_dum[x_dum['RUL']==j]['RUL'],c = ListedColormap(('green', 'red'))(i), label = j)
        ax.set_title('ID : '+str(iden[k]), fontsize=12,fontweight='bold')
        ax.set_ylim([-0.5,1.5])
        ax.set_xlabel('Cycles')
        ax.set_xlim(left=1)
        ax.set_xlim(right=ax.get_xlim()[1]+4)
        ax.set_yticks((0,1))
        ax.set_yticklabels(('OK','Caution'))
    k=k+1
fig.canvas.manager.full_screen_toggle()
plt.savefig('monitor.png')
plt.close(fig)

#%% Confusion matric plot

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)

plt.title('Confusion Matrix', fontsize=18)
plt.savefig('Confusion_Mat.png')
plt.close(fig)
# plt.show()

#%%
from matplotlib.colors import ListedColormap
data=X_2.join(pred,how='outer')
df=data[['ID','Cycle','RUL']]
x_dum=df[df['ID']==35].sort_values(by=['Cycle'])
fig, ax = plt.subplots(nrows=1,ncols=1,constrained_layout=True)
for i, j in enumerate(np.unique(data['RUL'])):
    ax.scatter(x_dum[x_dum['RUL']==j]['Cycle'],x_dum[x_dum['RUL']==j]['RUL'],c = ListedColormap(('green', 'red'))(i), label = j)
    ax.set_title('ID : '+str(35), fontsize=20,fontweight='bold')
    ax.set_ylim([-0.5,1.5])
    ax.set_xlabel('Cycles')
    ax.set_xlim(left=1)
    ax.set_xlim(right=ax.get_xlim()[1]+4)
    ax.set_yticks((0,1))
    ax.set_yticklabels(('OK','Caution'))
fig.savefig('Fig1.png')
plt.close(fig)
