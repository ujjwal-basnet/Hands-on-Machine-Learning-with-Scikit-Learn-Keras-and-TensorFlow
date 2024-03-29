import basic 
import featureScalling


df = basic.load_housing_data()




def log_scaled_plot(feature):
    log_scaled = featureScalling.log_scale(feature) 
    basic.sns.histplot(x = log_scaled)
    basic.plt.xlabel(f"Log Scaled {feature}")
    basic.plt.ylabel("Frequency")
    basic.plt.show()
    basic.plt.savefig(f'histogra_plot_of_{feature.name}')

    

def plot_hist(feature):
    basic.sns.histplot(feature)
    basic.plt.xlabel(f"{feature}")
    basic.plt.ylabel("Frequency")
    basic.plt.show()
    basic.plt.savefig(f'histogra_plot_png')
    
    
def compare_plots(plt1 , plt2):
    fig , (ax1 , ax2) = basic.plt.subplots()
    
    
    
    
    
    

plot_hist(df['population'])
log_scaled_plot(df['population'])

    
    
    
    
    
    




