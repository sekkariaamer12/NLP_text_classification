import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import ticker
from torch import nn
from torch.optim import Adam
from torch.testing._internal.common_quantization import accuracy
from torchtext.vocab import build_vocab_from_iterator
import torch.nn.functional as F
import time
from RNN import RNN
from torch.utils.data import Dataset, DataLoader, TensorDataset

parameters = {
    "BATCH_SIZE": 128,
    "LR": 0.0009,
    "EPOCHS":20
}

"""
Cette fonction lit un fichier et retourne la liste des textes et la liste des émotions
"""


def load_file(file):
    phrases_split = []
    phrases = []
    emotions = []
    with open(file, "r") as f:
        lignes = f.readlines()
        for ligne in lignes:
            ligne = ligne.split(";")
            # On fait des listes de listes pour pouvoir utiliser le build vocab
            phrases_split.append(ligne[0].split())
            phrases.append(ligne[0])
            emotions.append(ligne[1].strip("\n").split())

    return phrases_split, emotions


def create_vocab(phrases):
    return build_vocab_from_iterator(phrases)


def return_vocab(phrases, vocab):
    resultat = []
    for phrase in phrases:
        resultat.append(np.array(vocab(phrase)))
    return resultat



class encode():
    def __init__(self, numClass):
        self.numClass = numClass
        self.encoder = F.one_hot

    def make_encodage(self, line):
        return self.encoder(torch.tensor(line, device=device).long(), num_classes=self.numClass)


if __name__ == '__main__':
    torch.cuda.empty_cache()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # On load les différents set de données
    phrases_train, emotion_train = load_file("dataset/train.txt")
    phrases_test, emotion_test = load_file("dataset/test.txt")
    phrases_val, emotion_val = load_file("dataset/val.txt")


    # On réduit la taille des phrases à 10 en ne gardant que les 10 premiers mots
    for k in range (len(phrases_train)):
        del phrases_train[k][40:]

        # On rajoute des caractères vides en début de phrase si sa longueur n'est pas égale à 10
        if len(phrases_train[k]) != 40:
            liste = [''] * (40 - len(phrases_train[k]))
            phrases_train[k] = liste + phrases_train[k]

    for k in range (len(phrases_val)):
        del phrases_val[k][40:]
        if len(phrases_val[k]) != 40:
            liste = [''] * (40 - len(phrases_val[k]))
            phrases_val[k] = liste + phrases_val[k]

    for k in range (len(phrases_test)):
        del phrases_test[k][40:]
        if len(phrases_test[k]) != 40:
            liste = [''] * (40 - len(phrases_test[k]))
            phrases_test[k] = liste + phrases_test[k]
    # on créer le vocabulaire contenant l ensemble des mots présents dans les trois dataset
    voc_phrases = create_vocab(phrases_train + phrases_val + phrases_test)

    # On créer le vocabulaire contenant l ensemble des emotions
    voc_emotions = create_vocab(emotion_train + emotion_val + emotion_test)

    # On convertit les phrases avec leur identifiant contenu dans voc_phrases
    phrases_convert_to_id = return_vocab(phrases_train, voc_phrases)
    phrases_val_id = return_vocab(phrases_val, voc_phrases)
    phrases_test_id = return_vocab(phrases_test, voc_phrases)
#    phrases_validate_id = return_vocab(phrases_val, voc_phrases)
    # print(phrases_convert_to_id)

    # On convertit les emotions avec leur identifiants contenu dans voc_emotions
    emotions_convert_to_id = return_vocab(emotion_train, voc_emotions)
    emotions_val_id = return_vocab(emotion_val, voc_emotions)
    emotions_test_id = return_vocab(emotion_test, voc_emotions)
    #emotions_validate_id = return_vocab(emotion_val, voc_emotions)

    # On test l encodage one hot sur les phrases
    encoder_phrases = encode(len(voc_phrases))
    # print(phrases_convert_to_id[0][0])

    model = RNN(len(voc_phrases), 16, len(voc_emotions), 16, device)
    criterion = nn.NLLLoss()

    # Define optimizer
    optimizer = Adam(model.parameters(), lr=parameters['LR'])

    train_dataset = TensorDataset(torch.tensor(np.array(phrases_convert_to_id), device=device), torch.tensor(np.array(emotions_convert_to_id), device=device))
    val_dataset = TensorDataset(torch.tensor(np.array(phrases_val_id), device=device),
                                  torch.tensor(np.array(emotions_val_id), device=device))

    test_dataset = TensorDataset(torch.tensor(np.array(phrases_test_id), device=device),
                                torch.tensor(np.array(emotions_test_id), device=device))

    train_dataset = DataLoader(train_dataset, batch_size=parameters['BATCH_SIZE'], shuffle=True)

    val_dataset = DataLoader(val_dataset, batch_size=parameters['BATCH_SIZE'], shuffle=True)
    test_dataset = DataLoader(test_dataset, 1, shuffle=True)


    y_train = []
    print_every = 20
    clip = 5
    counter = 0
    loss_train = []
    loss_valid = []
    acc_train = []
    acc_val = []
    for epoch in range(parameters['EPOCHS']):
        start = time.time()
        correct = 0
        total = 0
        ### Initialize hidden state
        compteur = 0
        loss_train_epoch = []
        liste_acc = []
        ### Training
        for inp, labels in train_dataset:
            hidden = model.initHidden()

            data_train = encoder_phrases.make_encodage(inp)

            model.zero_grad()
            # Pour chaque valeur
            for k in range(len(inp[0])):
                output, hidden = model(data_train.permute(1, 0, 2)[k].float(), hidden)
            loss = criterion(output.squeeze(), labels.squeeze().long())
            loss_train_epoch.append(loss.item())
            loss.backward()
            optimizer.step()
            prediction = output.topk(1)[1].squeeze()
            for k in range(len(prediction)):
                if prediction[k] == labels[k].squeeze():
                    correct += 1
                total += 1
            liste_acc.append(correct / total)
            """for p in model.parameters():
                p.data.add_(p.grad.data, alpha=parameters['LR'])"""
            compteur += 1


            if compteur % 100 == 0:
                print('Epoch {} / {} : {} : LOSS = {}'.format(epoch + 1, parameters['EPOCHS'], compteur, loss))
        counter += 1
        y_train.append(counter)
        loss_train.append(sum(loss_train_epoch)/len(loss_train_epoch))
        acc_train.append(sum(liste_acc) / len(liste_acc))
        liste_acc = []
        correct = 0
        total = 0
        loss_valid_epoch = []

        # Validation
        for inp, labels in val_dataset:
            correct = 0
            total = 0

            compteur = 0

            if len(inp) == parameters['BATCH_SIZE']:
                hidden = model.initHidden()
                data_val = encoder_phrases.make_encodage(inp)


                for k in range(len(inp[0])):
                    output, hidden = model(data_val.permute(1, 0, 2)[k].float(), hidden)
                loss_val = criterion(output.squeeze(), labels.squeeze().long())
                loss_valid_epoch.append(loss_val.item())
                prediction = output.topk(1)[1].squeeze()
                for k in range(len(prediction)):
                    if prediction[k] == labels[k].squeeze():
                        correct += 1
                    total += 1
                liste_acc.append(correct/total)
                compteur += 1
        loss_valid.append(sum(loss_valid_epoch)/len(loss_valid_epoch))
        acc_val.append(sum(liste_acc)/len(liste_acc))
        print ("Accuracy Validation : {}".format(sum(liste_acc)/len(liste_acc)))

    print(counter)
    print(loss_train)
    plt.title("Loss")
    plt.plot(y_train, loss_train, label="Training")
    plt.plot(y_train, loss_valid, label="Validation")

    plt.show()

    plt.title("Accuracy")
    plt.plot(y_train, acc_train, label='Training')
    plt.plot(y_train, acc_val, label="Validation")

    plt.show()

    # Test
    confusion = torch.zeros(len(voc_emotions), len(voc_emotions))
    for inp, labels in train_dataset:
        hidden = model.initHidden()

        data_train = encoder_phrases.make_encodage(inp)

        model.zero_grad()
        # Pour chaque valeur
        for k in range(len(inp[0])):
            output, hidden = model(data_train.permute(1, 0, 2)[k].float(), hidden)
        prediction = output.topk(1)[1].squeeze()
        confusion[labels.squeeze()[0]][prediction.squeeze().int()[0]] += 1

    for i in range(len(voc_emotions)):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    all_categories = ['sadness', 'anger', 'love', 'surprise', 'fear', 'joy']
    ax.set_xticklabels([''] +all_categories, rotation=90)
    ax.set_yticklabels([''] +all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()
