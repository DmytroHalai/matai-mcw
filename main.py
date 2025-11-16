import numpy as np
import tkinter as tk
from tkinter import messagebox, Toplevel, scrolledtext
import matplotlib.pyplot as plt
from genetic_algorithm import StructuralGeneticAlgorithm
from utils import decode
from network import NeuralNetwork


def run_structural_ga():
    try:
        func_str = entry_func.get()
        a = float(entry_from.get())
        b = float(entry_to.get())
        max_neurons = int(entry_max_neurons.get())
        max_error = float(entry_max_error.get())
        generations = int(entry_gen.get())
        pop_size = int(entry_pop.get())
        mutation_rate = float(entry_mut.get())

        X = np.linspace(a, b, 30)
        Y = np.linspace(a, b, 30)
        X, Y = np.meshgrid(X, Y)
        Z_real = eval(func_str)

        info = Toplevel(root)
        info.title("Інформація про роботу ГА")
        log = scrolledtext.ScrolledText(info, width=60, height=30)
        log.pack()

        def log_callback(gen, structure, err):
            log.insert(tk.END, f"Покоління {gen} | Структура: {structure} | Похибка: {err:.5f}\n")
            log.see(tk.END)

        ga = StructuralGeneticAlgorithm(
            X, Y, Z_real,
            pop_size=pop_size,
            mutation_rate=mutation_rate,
            max_neurons=max_neurons,
            max_error=max_error
        )

        best, history = ga.evolve(generations, info_callback=log_callback)

        structure, best_genome = best

        nn = NeuralNetwork(structure)
        decode(nn, best_genome)

        inputs = np.vstack((X.ravel(), Y.ravel()))
        Z_pred = nn.forward(inputs).reshape(X.shape)

        messagebox.showinfo("Готово",
                            f"Структура знайденої мережі: {structure}\n"
                            f"Похибка: {history[-1][2]:.5f}")

        # 3D графіки
        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1.plot_surface(X, Y, Z_real, cmap="viridis")
        ax1.set_title("Реальна функція")

        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        ax2.plot_surface(X, Y, Z_pred, cmap="plasma")
        ax2.set_title("Синтезована мережа")
        plt.show()

    except Exception as e:
        messagebox.showerror("Помилка", str(e))


root = tk.Tk()
root.title("Структурний синтез НМ (МКР +20)")

tk.Label(root, text="Функція:").pack()
entry_func = tk.Entry(root, width=60)
entry_func.insert(0, "np.cos(np.abs(Y)) + np.sin(X + Y)")
entry_func.pack()

tk.Label(root, text="Відрізок від:").pack()
entry_from = tk.Entry(root)
entry_from.insert(0, "-2")
entry_from.pack()

tk.Label(root, text="Відрізок до:").pack()
entry_to = tk.Entry(root)
entry_to.insert(0, "2")
entry_to.pack()

tk.Label(root, text="Максимальна к-ть нейронів:").pack()
entry_max_neurons = tk.Entry(root)
entry_max_neurons.insert(0, "12")
entry_max_neurons.pack()

tk.Label(root, text="Максимальна похибка:").pack()
entry_max_error = tk.Entry(root)
entry_max_error.insert(0, "0.01")
entry_max_error.pack()

tk.Label(root, text="Покоління:").pack()
entry_gen = tk.Entry(root)
entry_gen.insert(0, "40")
entry_gen.pack()

tk.Label(root, text="Популяція:").pack()
entry_pop = tk.Entry(root)
entry_pop.insert(0, "25")
entry_pop.pack()

tk.Label(root, text="Мутація:").pack()
entry_mut = tk.Entry(root)
entry_mut.insert(0, "0.15")
entry_mut.pack()

tk.Button(root, text="Запустити структурний синтез", command=run_structural_ga).pack(pady=10)

root.mainloop()
