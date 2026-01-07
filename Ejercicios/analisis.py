import pandas as pd
import seaborn as sns
import re
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:
    df = pd.read_csv("datasetTexto.csv", sep=",", quotechar='"', engine="python")
except FileNotFoundError:
    exit()

df['Categoria'] = df['Categoria'].astype(str).str.strip().str.title()
df['Categoria'] = df['Categoria'].replace(['Nan', 'None', 'nan'], 'Sin Categoría')
df['Fecha'] = pd.to_datetime(df['Fecha'])

fecha_inicio = '2025-10-15'
df_filtrado = df[df['Fecha'] >= fecha_inicio].copy()

if df_filtrado.empty:
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("Error", "No hay datos disponibles para el rango de fecha seleccionado.")
    exit()

hashtags = df_filtrado['Comentario_Reaccion'].dropna().apply(lambda x: re.findall(r"#\w+", x))
hashtags_flat = [tag for sublist in hashtags for tag in sublist]
hashtags_series = pd.Series(hashtags_flat).value_counts()
hashtags_list = list(zip(hashtags_series.index, hashtags_series.values))

root = tk.Tk()
root.title("Dashboard")
root.state('zoomed')
root.configure(bg="#f0f0f0")

root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)
root.rowconfigure(0, weight=1)
root.rowconfigure(1, weight=1)

frame1 = tk.Frame(root, bg="white", bd=2, relief="groove")
frame1.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

fig1 = Figure(figsize=(5, 4), dpi=100)
ax1 = fig1.add_subplot(111)

conteo = df_filtrado['Categoria'].value_counts()
if not conteo.empty:
    bars = ax1.bar(conteo.index, conteo.values, color='skyblue')
    ax1.bar_label(bars, fmt='%d', padding=3)
    ax1.set_title("Artículos por Categoría")
    ax1.tick_params(axis='x', rotation=45, labelsize=8)
    fig1.subplots_adjust(bottom=0.3)

canvas1 = FigureCanvasTkAgg(fig1, master=frame1)
canvas1.draw()
canvas1.get_tk_widget().pack(fill="both", expand=True)

frame2 = tk.Frame(root, bg="white", bd=2, relief="groove")
frame2.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

fig2 = Figure(figsize=(5, 4), dpi=100)
ax2 = fig2.add_subplot(111)

conteo_fecha = df_filtrado.groupby(['Fecha', 'Categoria']).size().reset_index(name='conteo')
if not conteo_fecha.empty:
    sns.lineplot(data=conteo_fecha, x='Fecha', y='conteo', hue='Categoria', marker="o", ax=ax2)
    ax2.set_title("Evolución Temporal")
    ax2.tick_params(axis='x', rotation=45, labelsize=8)
    ax2.legend(fontsize='small')
    fig2.set_tight_layout(True)

canvas2 = FigureCanvasTkAgg(fig2, master=frame2)
canvas2.draw()
canvas2.get_tk_widget().pack(fill="both", expand=True)

frame3 = tk.Frame(root, bg="white", bd=2, relief="groove")
frame3.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

fig3 = Figure(figsize=(5, 4), dpi=100)
ax3 = fig3.add_subplot(111)

top_medios = df_filtrado['Medio'].value_counts().head(10)
if not top_medios.empty:
    top_medios.plot(kind='barh', color='salmon', ax=ax3)
    ax3.set_title("Top 10 Medios")
    ax3.invert_yaxis()
    fig3.set_tight_layout(True)

canvas3 = FigureCanvasTkAgg(fig3, master=frame3)
canvas3.draw()
canvas3.get_tk_widget().pack(fill="both", expand=True)

frame4 = tk.Frame(root, bg="white", bd=2, relief="groove")
frame4.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

lbl = tk.Label(frame4, text="Top Hashtags", font=("Arial", 12, "bold"), bg="white")
lbl.pack(pady=5)

tree = ttk.Treeview(frame4, columns=("Hashtag", "Frecuencia"), show="headings")
tree.heading("Hashtag", text="Hashtag")
tree.heading("Frecuencia", text="Frecuencia")
tree.column("Hashtag", width=200)
tree.column("Frecuencia", width=100, anchor="center")

scrollbar = ttk.Scrollbar(frame4, orient="vertical", command=tree.yview)
tree.configure(yscroll=scrollbar.set)
tree.pack(side="left", fill="both", expand=True, padx=10, pady=10)
scrollbar.pack(side="right", fill="y", pady=10)

for h, f in hashtags_list[:20]:
    tree.insert("", "end", values=(h, f))

root.mainloop()