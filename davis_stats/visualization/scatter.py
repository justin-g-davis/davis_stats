import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def scatter(df, y, x, fit_line=False, dpi=150, figsize=(6, 4)):
    """
    Create a 2D or 3D scatter plot with optional fit line.
    """
    
    # Convert x to list if string
    if isinstance(x, str):
        x = [x]
    
    # Determine plot type based on number of x variables
    if len(x) == 1:
        # 2D scatter plot
        x_var = x[0]
        
        # Calculate correlation coefficient
        corr = df[x_var].corr(df[y])
        
        # Set style
        sns.set_style("whitegrid")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Create scatter plot
        if fit_line:
            sns.regplot(data=df, 
                       x=x_var, 
                       y=y,
                       scatter_kws={'alpha':0.5},
                       line_kws={'color': 'red'},
                       ci=None)
        else:
            sns.scatterplot(data=df,
                           x=x_var,
                           y=y,
                           alpha=0.5)
        
        # Customize plot
        plt.title(f'{y} and {x_var}\nCorrelation: {corr:.3f}', pad=15)
        plt.xlabel(x_var)
        plt.ylabel(y)
        
        # Adjust layout
        plt.tight_layout()
        
    else:
        # 3D scatter plot (use first 2 variables from x list)
        x_var = x[0]
        z_var = x[1]
        
        # Create 3D figure
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Remove NaN values for plotting
        plot_df = df[[x_var, z_var, y]].dropna()
        
        # Create 3D scatter plot
        scatter = ax.scatter(plot_df[x_var], plot_df[z_var], plot_df[y], 
                            c=plot_df[y], cmap='RdYlBu_r', s=50,
                            edgecolor='black', linewidth=0.5,
                            alpha=0.8)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.15, shrink=0.8)
        cbar.set_label(y, rotation=270, labelpad=15)
        
        # Add best-fit plane if requested
        if fit_line:
            # Prepare data for plane fitting: y = a*x + b*z + c
            X_data = np.column_stack([plot_df[x_var], plot_df[z_var], np.ones(len(plot_df))])
            y_data = plot_df[y].values
            
            # Fit plane using least squares
            coeffs, residuals, rank, s = np.linalg.lstsq(X_data, y_data, rcond=None)
            a, b, c = coeffs
            
            # Create mesh grid for the regression plane
            x_surf = np.linspace(plot_df[x_var].min(), plot_df[x_var].max(), 20)
            z_surf = np.linspace(plot_df[z_var].min(), plot_df[z_var].max(), 20)
            X_mesh, Z_mesh = np.meshgrid(x_surf, z_surf)
            
            # Calculate y values for the plane
            Y_mesh = a * X_mesh + b * Z_mesh + c
            
            # Plot the regression plane
            surf = ax.plot_surface(X_mesh, Z_mesh, Y_mesh, 
                                   alpha=0.4, cmap='coolwarm',
                                   edgecolor='black', linewidth=0.5,
                                   rstride=1, cstride=1,
                                   antialiased=True)
        
        # Set title
        title = f'{y}, {x_var}, and {z_var}'
        
        # Set axis labels
        ax.set_xlabel(x_var, labelpad=10)
        ax.set_ylabel(z_var, labelpad=10)
        ax.set_zlabel(y, labelpad=10)
        
        # Move z-axis to left side
        ax.zaxis._axinfo['juggled'] = (1, 2, 0)
        
        # Flip axes
        ax.invert_xaxis()
        ax.invert_yaxis()
        
        # Adjust viewing angle
        ax.view_init(elev=25, azim=135)
        
        # Grid styling
        ax.xaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.5)
        ax.yaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.5)
        ax.zaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.5)
        
        # Set title
        ax.set_title(title, pad=15)
        
        # Adjust layout
        plt.tight_layout()
    
    # Show plot
    plt.show()
