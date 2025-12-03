"""
Plotting and Visualization for Single Microresonator Simulations
================================================================
Generates publication-quality figures and interactive plots

Author: DTU Photonics
Date: November 2025
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.io import loadmat
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.linewidth'] = 1.0

# Try to import ipywidgets for interactive plots
try:
    from ipywidgets import interact, widgets
    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False
    print("⚠️  ipywidgets not installed - interactive plots disabled")
    print("   Install with: pip install ipywidgets")

# ============================================
# LOAD DATA
# ============================================
print("📊 Loading simulation results...")
data = loadmat('SSFM_half_data_optimized.mat')

u_probe = data['u_probe']
driving_force = data['driving_force']
detuning = data['detuning'].flatten()
t_sim = data['t_sim'].flatten()
kext = data['kappa_ext'].item() if data['kappa_ext'].size == 1 else data['kappa_ext'][0, 0]
alpha = data['alpha'].item() if data['alpha'].size == 1 else data['alpha'][0, 0]
k0 = data['kappa_0'].item() if data['kappa_0'].size == 1 else data['kappa_0'][0, 0]

print(f"   Probes: {u_probe.shape[0]}")
print(f"   Modes: {u_probe.shape[1]}")
print(f"   κ_ext: {kext:.3e}")
print(f"   α_total: {alpha:.3e}\n")

# Load dispersion for frequency axis
disp = loadmat('disp.mat')
FSR = disp['FSR'][0, 0] if disp['FSR'].size == 1 else disp['FSR'][0]
sim = loadmat('sim.mat')
fpmp = sim['f_pmp'][0, 0] if sim['f_pmp'].size == 1 else sim['f_pmp'][0]

# ============================================
# COMPUTE OUTPUT FIELDS
# ============================================
print("🔧 Computing output fields...")

# Waveguide and cavity output fields
wg = driving_force * np.sqrt(1 - kext)
cav = np.sqrt(kext) * u_probe * np.exp(1j * np.pi)

# Time domain (normalized)
Awg = (wg + cav) / np.sqrt(u_probe.shape[1])
Acav = np.sqrt(alpha / 2) * u_probe * np.exp(1j * np.pi) / np.sqrt(u_probe.shape[1])

# Frequency domain
Ewg = np.fft.fftshift(np.fft.fft(Awg, axis=1), axes=1) / np.sqrt(u_probe.shape[1]) + 1e-12
Ecav = np.fft.fftshift(np.fft.fft(Acav, axis=1), axes=1) / np.sqrt(u_probe.shape[1])

# Power calculations
Pcomb = np.sum(np.abs(Ecav)**2, axis=1) - np.abs(Ecav[:, u_probe.shape[1]//2])**2
Pwg = np.sum(np.abs(Ewg)**2, axis=1)
Pcav = np.sum(np.abs(Ecav)**2, axis=1)

# Axes
detuning_GHz = detuning * 1e-9 / (2 * np.pi)
modes = np.arange(-(u_probe.shape[1] - 1) // 2, u_probe.shape[1] // 2 + 1, 1)
freq = fpmp + modes * FSR
tR_ps = 1e12 / FSR  # Round-trip time in ps

print("   ✓ Field calculations complete\n")

# ============================================
# FIGURE 1: FIELD EVOLUTION
# ============================================

def plot_field_evolution():
    """Plot spatiotemporal evolution"""
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    # Intracavity field amplitude
    ax1 = fig.add_subplot(gs[0, :])
    im1 = ax1.imshow(np.abs(Acav).T, aspect='auto', cmap='viridis',
                     origin='lower', interpolation='bilinear')
    ax1.set_xlabel('Detuning Step', fontsize=11)
    ax1.set_ylabel('Mode Index', fontsize=11)
    ax1.set_title('Intracavity Field Evolution |A(μ,t)|', fontweight='bold', fontsize=12)
    cbar1 = plt.colorbar(im1, ax=ax1, label='|A| (√W)', fraction=0.046, pad=0.02)
    
    # Spectral evolution
    ax2 = fig.add_subplot(gs[1, 0])
    im2 = ax2.imshow(np.abs(Ecav).T, aspect='auto', cmap='plasma',
                     origin='lower', interpolation='bilinear')
    ax2.set_xlabel('Detuning Step', fontsize=11)
    ax2.set_ylabel('Comb Line μ', fontsize=11)
    ax2.set_title('Frequency Comb Evolution', fontweight='bold', fontsize=11)
    cbar2 = plt.colorbar(im2, ax=ax2, label='|E_μ|', fraction=0.046, pad=0.02)
    
    # Intracavity power
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(detuning_GHz, Pcomb * 1e3, linewidth=2, color='#2E86AB', alpha=0.8)
    ax3.set_xlabel('Detuning (GHz)', fontsize=11)
    ax3.set_ylabel('Power (mW)', fontsize=11)
    ax3.set_title('Intracavity Comb Power', fontweight='bold', fontsize=11)
    ax3.grid(True, alpha=0.3, linestyle=':')
    
    plt.suptitle('Microresonator Soliton Formation', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig1_field_evolution_single.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: fig1_field_evolution_single.png")
    plt.show()

# ============================================
# FIGURE 2: FREQUENCY COMB SPECTRUM
# ============================================

def plot_comb_spectrum(idx=-1, xaxis='freq'):
    """Plot frequency comb with stem lines"""
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Prepare data
    if xaxis == 'modes':
        x = modes
        xlabel = 'Comb Line Number μ'
    elif xaxis == 'freq':
        x = freq * 1e-12
        xlabel = 'Frequency (THz)'
    
    # Create stem plot effect
    x_stems = np.zeros(x.size * 3)
    x_stems[::3] = x
    x_stems[1::3] = x
    x_stems[2::3] = x
    
    y = 10 * np.log10(np.abs(Ecav[idx])**2) + 30
    y_stems = np.zeros(y.size * 3)
    floor = -100
    y_stems[::3] = floor
    y_stems[1::3] = y
    y_stems[2::3] = floor
    
    # Plot stems
    ax.vlines(x_stems[::3], ymin=floor, ymax=y_stems[1::3],
              colors='#2E86AB', linestyles='-', linewidth=1.5, alpha=0.7)
    ax.scatter(x, y, color='#2E86AB', s=20, zorder=3)
    
    # Styling
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel('Power (dBm)', fontsize=12, fontweight='bold')
    ax.set_title(f'Frequency Comb Spectrum (Detuning: {detuning_GHz[idx]:.2f} GHz)',
                 fontsize=13, fontweight='bold')
    ax.set_ylim([y.max() - 80, y.max() + 5])
    ax.grid(True, alpha=0.3, linestyle=':')
    
    # Add -3dB line
    ax.axhline(y=y.max() - 3, color='red', linestyle='--',
               linewidth=1, alpha=0.5, label='-3 dB')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'fig2_comb_spectrum_idx{idx}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: fig2_comb_spectrum_idx{idx}.png")
    plt.show()

# ============================================
# FIGURE 3: TEMPORAL SOLITON PROFILE
# ============================================

def plot_soliton_time(idx=-1):
    """Plot soliton pulse in time domain"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    t = np.linspace(-tR_ps / 2, tR_ps / 2, len(Ecav[idx]))
    
    # Power
    ax1.plot(t, np.abs(Acav[idx])**2 * 1e3, linewidth=2, color='#2E86AB')
    ax1.set_ylabel('Power (mW)', fontsize=12, fontweight='bold')
    ax1.set_title('Soliton Temporal Profile', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle=':')
    
    # Amplitude and phase
    ax2_twin = ax2.twinx()
    line1 = ax2.plot(t, np.abs(Acav[idx]), linewidth=2, color='#2E86AB',
                     label='Amplitude |A|', alpha=0.8)
    line2 = ax2_twin.plot(t, np.angle(Acav[idx]), linewidth=2, color='#F18F01',
                          linestyle='--', label='Phase ∠A', alpha=0.7)
    
    ax2.set_xlabel('Time (ps)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Amplitude (√W)', fontsize=11, fontweight='bold', color='#2E86AB')
    ax2_twin.set_ylabel('Phase (rad)', fontsize=11, fontweight='bold', color='#F18F01')
    ax2.tick_params(axis='y', labelcolor='#2E86AB')
    ax2_twin.tick_params(axis='y', labelcolor='#F18F01')
    ax2.grid(True, alpha=0.3, linestyle=':')
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, fontsize=10, loc='upper right')
    
    plt.suptitle(f'Soliton at Detuning: {detuning_GHz[idx]:.2f} GHz',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'fig3_soliton_time_idx{idx}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: fig3_soliton_time_idx{idx}.png")
    plt.show()

# ============================================
# FIGURE 4: POWER DYNAMICS
# ============================================

def plot_power_dynamics():
    """Plot various power metrics vs detuning"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Total power
    axes[0].plot(detuning_GHz, Pcomb * 1e3, linewidth=2, color='#2E86AB', label='Comb')
    axes[0].plot(detuning_GHz, Pwg * 1e3, linewidth=2, color='#A23B72',
                 alpha=0.6, linestyle='--', label='Waveguide')
    axes[0].set_ylabel('Power (mW)', fontsize=11, fontweight='bold')
    axes[0].set_title('Total Power vs Detuning', fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, linestyle=':')
    
    # Peak power
    P_peak = np.max(np.abs(Acav)**2, axis=1)
    axes[1].plot(detuning_GHz, P_peak * 1e3, linewidth=2, color='#F18F01')
    axes[1].set_ylabel('Peak Power (mW)', fontsize=11, fontweight='bold')
    axes[1].set_title('Soliton Peak Power', fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle=':')
    
    # Pulse width (FWHM approximation)
    fwhm = []
    for i in range(len(Acav)):
        p = np.abs(Acav[i])**2
        fwhm.append(np.sum(p > p.max() / 2))
    
    axes[2].plot(detuning_GHz, fwhm, linewidth=2, color='#5C7F67')
    axes[2].set_xlabel('Detuning (GHz)', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('FWHM (points)', fontsize=11, fontweight='bold')
    axes[2].set_title('Soliton Width', fontsize=11, fontweight='bold')
    axes[2].grid(True, alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    plt.savefig('fig4_power_dynamics.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: fig4_power_dynamics.png")
    plt.show()

# ============================================
# INTERACTIVE PLOT (JUPYTER)
# ============================================

if HAS_WIDGETS:
    def plot_interactive_single(idx):
        """Interactive plot for Jupyter notebooks"""
        fig, axes = plt.subplots(4, 1, figsize=(14, 12),
                                gridspec_kw={'height_ratios': [1, 1, 1, 1]})
        
        # Evolution map
        im = axes[0].imshow(np.abs(Acav).T, aspect='auto', cmap='viridis', origin='lower')
        axes[0].axvline(x=idx, color='red', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Detuning Step')
        axes[0].set_ylabel('Mode Index')
        axes[0].set_title("Intracavity Field Evolution")
        
        # Power vs detuning
        axes[1].plot(Pcomb * 1e3, linewidth=2, color='#2E86AB')
        axes[1].axvline(x=idx, color='red', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Power (mW)')
        axes[1].set_title('Intracavity Power')
        axes[1].grid(True, alpha=0.3)
        
        # Frequency comb
        x = freq * 1e-12
        y = 10 * np.log10(np.abs(Ecav[idx])**2) + 30
        axes[2].vlines(x, ymin=-100, ymax=y, colors='#2E86AB', linewidth=1.5)
        axes[2].scatter(x, y, color='#2E86AB', s=20)
        axes[2].set_xlabel('Frequency (THz)')
        axes[2].set_ylabel('Power (dBm)')
        axes[2].set_title(f'Spectrum (Detuning: {detuning_GHz[idx]:.2f} GHz)')
        axes[2].set_ylim([y.max() - 80, y.max() + 5])
        axes[2].grid(True, alpha=0.3)
        
        # Time domain
        t = np.linspace(-tR_ps / 2, tR_ps / 2, len(Acav[idx]))
        axes[3].plot(t, np.abs(Acav[idx])**2 * 1e3, linewidth=2, color='#2E86AB')
        axes[3].set_xlabel('Time (ps)')
        axes[3].set_ylabel('Power (mW)')
        axes[3].set_title('Temporal Soliton Profile')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    print("\n📊 Interactive plot available!")
    print("   Use: interact(plot_interactive_single, idx=widgets.IntSlider(...))")

# ============================================
# MAIN EXECUTION
# ============================================
# %%
if __name__ == "__main__":
    print("\n" + "="*60)
    print("GENERATING VISUALIZATION FIGURES")
    print("="*60 + "\n")
    
    print("Figure 1: Field Evolution...")
    plot_field_evolution()
    
    print("\nFigure 2: Frequency Comb Spectrum...")
    plot_comb_spectrum(idx=-1, xaxis='freq')
    
    print("\nFigure 3: Temporal Soliton Profile...")
    plot_soliton_time(idx=-1)
    
    print("\nFigure 4: Power Dynamics...")
    plot_power_dynamics()
    
    print("\n" + "="*60)
    print("✅ ALL FIGURES GENERATED SUCCESSFULLY")
    print("="*60)
    print("\nSaved files:")
    print("  - fig1_field_evolution_single.png")
    print("  - fig2_comb_spectrum_idx-1.png")
    print("  - fig3_soliton_time_idx-1.png")
    print("  - fig4_power_dynamics.png")
    
    if HAS_WIDGETS:
        print("\n📊 For interactive plots in Jupyter:")
        print("   from ipywidgets import interact, widgets")
        print("   slider = widgets.IntSlider(value=1000, min=0, max=len(Acav)-1)")
        print("   interact(plot_interactive_single, idx=slider)")
    
    print("\n🎉 Ready for your IEEE presentation!\n")
# %%
