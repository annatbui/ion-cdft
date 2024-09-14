import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import matplotlib.cm as cm

# Set up the color cycle
num_colors = 20
colors = cm.RdPu(np.linspace(0, 1, num_colors))
color_cycle = np.tile(colors, (int(np.ceil(1000000 / num_colors)), 1))[:1000000]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_cycle)

# Update matplotlib parameters
params = {"axes.labelsize": 14, "axes.titlesize": 15}
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'
plt.rcParams.update(params)
plt.rcParams['figure.dpi'] = 300


def configure_plot(zbins):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=True)
    ax[0].set_xlim(zbins[0], zbins[-1])
    ax[1].set_ylim(-17, 0)
    ax[0].set_ylabel(r'$\rho(z)$ [$\mathrm{\AA}^{-3}$]')
    ax[1].set_ylabel(r'$\beta[\mu - V_{\mathrm{ext}}(z) - q \phi_\mathrm{R}(z)]$')
    ax[0].set_xlabel(r'$z$ [$\mathrm{\AA}$]')
    ax[1].set_xlabel(r'$z$ [$\mathrm{\AA}$]')
    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.tick_right()
    
    ax[0].grid(which="major", ls="dashed", dashes=(1, 3), lw=0.5, zorder=0)
    ax[1].grid(which="major", ls="dashed", dashes=(1, 3), lw=0.5, zorder=0)
    ax[0].tick_params(direction="in", which="major", length=5, labelsize=13)
    ax[1].tick_params(direction="in", which="major", length=5, labelsize=13)
    
    plt.tight_layout()
    return fig, ax

def plot_interactive_SR_onetype(fig, ax, zbins, rho, muloc, color_count):
    display.clear_output(wait=True)
    ax[0].plot(zbins, rho, color=color_cycle[color_count])
    ax[1].plot(zbins, muloc, color=color_cycle[color_count])
    display.display(fig)

def plot_interactive_SR_twotype(fig, ax, zbins, rho_H, rho_O, muloc_H, muloc_O, color_count):
    display.clear_output(wait=True)
    ax[0].plot(zbins, rho_H, color=color_cycle[color_count] )
    ax[0].plot(zbins, rho_O, ls="--", color=color_cycle[color_count])
    ax[1].plot(zbins, muloc_H, color=color_cycle[color_count] )
    ax[1].plot(zbins, muloc_O, ls="--", color=color_cycle[color_count] )
    display.display(fig)

def plot_interactive_LR_onetype(fig, ax, zbins, rho, muloc, charge, lmf_z, color_count):
    display.clear_output(wait=True)
    ax[0].plot(zbins, rho, color=color_cycle[color_count])
    ax[1].plot(zbins, muloc - charge * lmf_z, color=color_cycle[color_count])
    ax[1].plot(zbins, charge * lmf_z, ls='--', color=color_cycle[color_count])
    display.display(fig)
    
def plot_end_SR_onetype(zbins, rho, muloc, ax):
    display.clear_output(wait=False)
    ax[0].plot(zbins, rho, color='black', lw=2)
    ax[1].plot(zbins, muloc, color='black', lw=2)
    
def plot_end_SR_twotype(zbins, rhoH, rhoO, mulocH, mulocO, ax):
    display.clear_output(wait=False)
    ax[0].plot(zbins, rhoH, color='black', lw=2)
    ax[0].plot(zbins, rhoO, ls='--', color='black', lw=2)
    ax[1].plot(zbins, mulocH, color='black', lw=2)
    ax[1].plot(zbins, mulocO, ls='--', color='black', lw=2)
    
def plot_end_LR_onetype(zbins, rho, muloc, charge, lmf_z, ax):
    display.clear_output(wait=False)
    ax[0].plot(zbins, rho, color='black', lw=2)
    ax[1].plot(zbins, muloc-charge*lmf_z, color='black', lw=2)
    ax[1].plot(zbins, charge*lmf_z, ls='--', color='black', lw=2)

