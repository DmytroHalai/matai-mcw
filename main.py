import numpy as np
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from genetic_algorithm import GeneticAlgorithm
from network import NeuralNetwork
from utils import decode

def run_ga():
    try:
        generations = int(entry_gen.get())
        pop_size = int(entry_pop.get())
        mutation_rate = float(entry_mut.get())
        func_str = entry_func.get()

        X = np.linspace(-2, 2, 30)
        Y = np.linspace(-2, 2, 30)
        X, Y = np.meshgrid(X, Y)
        Z_real = eval(func_str)

        layer_sizes = [2, 4, 8, 10, 8, 8, 1]

        ga = GeneticAlgorithm(layer_sizes, X, Y, Z_real, pop_size, mutation_rate)
        best_genome, errors = ga.evolve(generations)

        plt.plot(errors)
        plt.title("Еволюція похибки")
        plt.xlabel("Покоління")
        plt.ylabel("MSE")
        plt.show()

        nn = NeuralNetwork(layer_sizes)
        decode(nn, best_genome)
        inputs = np.vstack((X.ravel(), Y.ravel()))
        Z_pred = nn.forward(inputs).reshape(X.shape)

        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1.plot_surface(X, Y, Z_real, cmap="viridis")
        ax1.set_title("Реальна функція")

        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        ax2.plot_surface(X, Y, Z_pred, cmap="plasma")
        ax2.set_title("Найкраща нейронна мережа після GA")
        plt.show()

        messagebox.showinfo("Готово", "Навчання завершено!")

    except Exception as e:
        messagebox.showerror("Помилка", str(e))

root = tk.Tk()
root.title("Нейро-нечітке моделювання (МКР)")

tk.Label(root, text="Функція (наприклад np.cos(np.abs(Y)) + np.sin(X + Y))").pack()
entry_func = tk.Entry(root, width=60)
entry_func.insert(0, "np.cos(np.abs(Y)) + np.sin(X + Y)")
entry_func.pack()

tk.Label(root, text="Кількість поколінь:").pack()
entry_gen = tk.Entry(root)
entry_gen.insert(0, "40")
entry_gen.pack()

tk.Label(root, text="Розмір популяції:").pack()
entry_pop = tk.Entry(root)
entry_pop.insert(0, "20")
entry_pop.pack()

tk.Label(root, text="Ймовірність мутації:").pack()
entry_mut = tk.Entry(root)
entry_mut.insert(0, "0.15")
entry_mut.pack()

tk.Button(root, text="Запустити навчання", command=run_ga).pack(pady=10)
root.mainloop()
