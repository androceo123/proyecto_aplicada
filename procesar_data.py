import pandas as pd
import re
import csv
import numpy as np
import skfuzzy as fuzz
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import time

# Descargar los datos necesarios de NLTK
nltk.download("vader_lexicon")

# Cargar los datos
file_path = "test_data.csv"
data = pd.read_csv(file_path)

# Define a dictionary for common English abbreviations with variations
abbreviations = {
    "i'm": "i am",
    "im": "i am",
    "i m": "i am",
    "you're": "you are",
    "youre": "you are",
    "you re": "you are",
    "he's": "he is",
    "hes": "he is",
    "he s": "he is",
    "she's": "she is",
    "shes": "she is",
    "she s": "she is",
    "it's": "it is",
    "it s": "it is",
    "we're": "we are",
    "were": "we are",
    "we re": "we are",
    "they're": "they are",
    "theyre": "they are",
    "they re": "they are",
    "i've": "i have",
    "ive": "i have",
    "i ve": "i have",
    "you've": "you have",
    "youve": "you have",
    "you ve": "you have",
    "we've": "we have",
    "weve": "we have",
    "we ve": "we have",
    "they've": "they have",
    "theyve": "they have",
    "they ve": "they have",
    "i'll": "i will",
    "ill": "i will",
    "i ll": "i will",
    "you'll": "you will",
    "youll": "you will",
    "you ll": "you will",
    "he'll": "he will",
    "hell": "he will",
    "he ll": "he will",
    "she'll": "she will",
    "shell": "she will",
    "she ll": "she will",
    "we'll": "we will",
    "well": "we will",
    "we ll": "we will",
    "they'll": "they will",
    "theyll": "they will",
    "they ll": "they will",
    "can't": "cannot",
    "cant": "cannot",
    "can t": "cannot",
    "won't": "will not",
    "wont": "will not",
    "won t": "will not",
    "don't": "do not",
    "dont": "do not",
    "do nt": "do not",
    "doesn't": "does not",
    "doesnt": "does not",
    "does nt": "does not",
    "didn't": "did not",
    "didnt": "did not",
    "did nt": "did not",
    "isn't": "is not",
    "isnt": "is not",
    "is nt": "is not",
    "aren't": "are not",
    "arent": "are not",
    "are nt": "are not",
    "wasn't": "was not",
    "wasnt": "was not",
    "was nt": "was not",
    "weren't": "were not",
    "werent": "were not",
    "were nt": "were not",
    "hasn't": "has not",
    "hasnt": "has not",
    "has nt": "has not",
    "haven't": "have not",
    "havent": "have not",
    "have nt": "have not",
    "hadn't": "had not",
    "hadnt": "had not",
    "had nt": "had not",
    "wouldn't": "would not",
    "wouldnt": "would not",
    "would nt": "would not",
    "shouldn't": "should not",
    "shouldnt": "should not",
    "should nt": "should not",
    "couldn't": "could not",
    "couldnt": "could not",
    "could nt": "could not",
    "mustn't": "must not",
    "mustnt": "must not",
    "must nt": "must not",
    "let's": "let us",
    "lets": "let us",
    "let s": "let us",
    "that's": "that is",
    "thats": "that is",
    "that s": "that is",
    "who's": "who is",
    "whos": "who is",
    "who s": "who is",
    "what's": "what is",
    "whats": "what is",
    "what s": "what is",
    "here's": "here is",
    "heres": "here is",
    "here s": "here is",
    "there's": "there is",
    "theres": "there is",
    "there s": "there is",
    "where's": "where is",
    "wheres": "where is",
    "where s": "where is",
    "how's": "how is",
    "hows": "how is",
    "how s": "how is",
    "y'all": "you all",
}


# Función para limpiar el texto
def clean_text(text):
    # Eliminar URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Eliminar menciones (@)
    text = re.sub(r"@\w+", "", text)

    # Eliminar puntuación y números
    text = re.sub(r"[^\w\s]|[\d]", " ", text)

    # Reemplazar múltiples espacios por un solo espacio
    text = re.sub(r"\s+", " ", text).strip()

    # Convertir a minúsculas y eliminar caracteres no ASCII
    text = text.lower().encode("ascii", "ignore").decode()

    return text

# Función para fuzzificar un valor basado en funciones de membresía
def fuzzify(value, membership_functions, value_range):
    memberships = [
        fuzz.interp_membership(value_range, mf, value) for mf in membership_functions
    ]
    return [float(m) for m in memberships]

# Función para agregar las salidas
def aggregate_outputs(rule_activations):
    neg_activation = np.fmax(rule_activations['R4'], 
                             np.fmax(rule_activations['R7'], rule_activations['R8']))
    neu_activation = np.fmax(rule_activations['R1'], 
                             np.fmax(rule_activations['R5'], rule_activations['R9']))
    pos_activation = np.fmax(rule_activations['R2'], 
                             np.fmax(rule_activations['R3'], rule_activations['R6']))
    
    # Recortar las funciones de membresía
    output_neg = np.fmin(neg_activation, output_negative)
    output_neu = np.fmin(neu_activation, output_neutral)
    output_pos = np.fmin(pos_activation, output_positive)
    
    # Combinar las salidas (agregación)
    aggregated_output = np.fmax(output_neg, np.fmax(output_neu, output_pos))
    
    return aggregated_output

# Función para la defuzzificación
def defuzzify_output(aggregated_output, x_output):
    # Calcular el centroide
    coa = fuzz.defuzz(x_output, aggregated_output, 'centroid')
    return coa

# Función para clasificar el sentimiento basado en el puntaje defuzzificado
def classify_sentiment(coa):
    if 0 <= coa < 3.3:
        return 'Negative'
    elif 3.3 <= coa < 6.7:
        return 'Neutral'
    elif 6.7 <= coa <= 10:
        return 'Positive'

# Inicializar el analizador de intensidad de sentimiento de NLTK
sia = SentimentIntensityAnalyzer()

# Limpiar el texto 
data["processed_sentence"] = data["sentence"].apply(clean_text)

# Calcular los puntajes de sentimiento 
for index, row in data.iterrows():
    time_score = time.time()
    sentiment_scores = sia.polarity_scores(row["processed_sentence"])
    time_score = time.time() - time_score
    data.at[index, "positive_score"] = sentiment_scores["pos"]
    data.at[index, "negative_score"] = sentiment_scores["neg"]
    data.at[index, "time_score"] = time_score

# Configuración de la fuzzificación
positive_max = data["positive_score"].max()
negative_max = data["negative_score"].max()
x_positive = np.linspace(0, positive_max, 100)
x_negative = np.linspace(0, negative_max, 100)

positive_low = fuzz.trimf(x_positive, [0, 0, positive_max / 2])
positive_medium = fuzz.trimf(x_positive, [0, positive_max / 2, positive_max])
positive_high = fuzz.trimf(x_positive, [positive_max / 2, positive_max, positive_max])

negative_low = fuzz.trimf(x_negative, [0, 0, negative_max / 2])
negative_medium = fuzz.trimf(x_negative, [0, negative_max / 2, negative_max])
negative_high = fuzz.trimf(x_negative, [negative_max / 2, negative_max, negative_max])

# Definir las funciones de membresía de salida
x_output = np.linspace(0, 10, 100)
output_positive = fuzz.trimf(x_output, [5, 10, 10])
output_neutral = fuzz.trimf(x_output, [0, 5, 10])
output_negative = fuzz.trimf(x_output, [0, 0, 5])

# Ejecutar todo el proceso y medir el tiempo
results = []
total_time = time.time()
for index, row in data.iterrows():
    start_time = time.time()  

    # Fuzzificar los puntajes de sentimiento positivos y negativos
    positive_fuzzy = fuzzify(row["positive_score"], [positive_low, positive_medium, positive_high], x_positive)
    negative_fuzzy = fuzzify(row["negative_score"], [negative_low, negative_medium, negative_high], x_negative)

    # Evaluar las reglas
    rule_activation = {
        'R1': np.fmin(positive_fuzzy[0], negative_fuzzy[0]),  # Low-Low -> Neutral
        'R2': np.fmin(positive_fuzzy[1], negative_fuzzy[0]),  # Medium-Low -> Positive
        'R3': np.fmin(positive_fuzzy[2], negative_fuzzy[0]),  # High-Low -> Positive
        'R4': np.fmin(positive_fuzzy[0], negative_fuzzy[1]),  # Low-Medium -> Negative
        'R5': np.fmin(positive_fuzzy[1], negative_fuzzy[1]),  # Medium-Medium -> Neutral
        'R6': np.fmin(positive_fuzzy[2], negative_fuzzy[1]),  # High-Medium -> Positive
        'R7': np.fmin(positive_fuzzy[0], negative_fuzzy[2]),  # Low-High -> Negative
        'R8': np.fmin(positive_fuzzy[1], negative_fuzzy[2]),  # Medium-High -> Negative
        'R9': np.fmin(positive_fuzzy[2], negative_fuzzy[2]),  # High-High -> Neutral
    }

    # Agregar las salidas
    aggregated_output = aggregate_outputs(rule_activation)

    # Defuzzificar para obtener el puntaje de sentimiento
    coa = defuzzify_output(aggregated_output, x_output)

    # Clasificar el sentimiento
    sentiment_class = classify_sentiment(coa)

    end_time = time.time() 
    execution_time = end_time - start_time + row["time_score"]

    # Agregar los resultados
    results.append({
        "Oración original": row["sentence"],
        "Label original": row["sentiment"],
        "Puntaje positivo": row["positive_score"],
        "Puntaje negativo": row["negative_score"],
        "COA": coa,
        "Resultado de inferencia": sentiment_class,
        "Tiempo de ejecución": execution_time
    })

total_time = time.time() - total_time
# Convertir los resultados a un DataFrame
results_df = pd.DataFrame(results)

# Calcular el tiempo de ejecución promedio
average_execution_time = results_df["Tiempo de ejecución"].mean()

# Guardar los resultados en un archivo CSV
results_df.to_csv("resultados_analisis_sentimiento.csv", index=False)

# Imprimir resumen del benchmark
print(f"Tiempo total de ejecución: {total_time:.6f} segundos")
print(f"Tiempo de ejecución promedio: {average_execution_time:.6f} segundos")
positive_count = len(results_df[results_df["Resultado de inferencia"] == "Positive"])
neutral_count = len(results_df[results_df["Resultado de inferencia"] == "Neutral"])
negative_count = len(results_df[results_df["Resultado de inferencia"] == "Negative"])
print(f"Total de tweets positivos: {positive_count}")
print(f"Total de tweets neutrales: {neutral_count}")
print(f"Total de tweets negativos: {negative_count}")
positive_time = results_df[results_df["Resultado de inferencia"] == "Positive"]["Tiempo de ejecución"].sum()
neutral_time = results_df[results_df["Resultado de inferencia"] == "Neutral"]["Tiempo de ejecución"].sum()
negative_time = results_df[results_df["Resultado de inferencia"] == "Negative"]["Tiempo de ejecución"].sum()
print(f"Tiempo total para tweets positivos: {positive_time:.6f} segundos")
print(f"Tiempo total para tweets neutrales: {neutral_time:.6f} segundos")
print(f"Tiempo total para tweets negativos: {negative_time:.6f} segundos")
