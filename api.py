# app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import faiss
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Chatbot Sportaabe API")
@app.get("/")  # Définition de la route principale
def read_root():
    return {"message": "Bienvenue sur mon API FastAPI!"}
# ---------------------------
# Configuration et initialisation

# Base de connaissances pour Sporttabe
documents = [
    """sportaabe est une application conçue pour digitaliser les offres de sport, bien-être et santé. Elle permet de réunir les acteurs du domaine
    organiser des activités, géolocaliser les offres existantes, informer les utilisateurs, partager les coûts, socialiser les utilisateurs et analyser
    les données. Elle a pour objectif de promouvoir la pratique sportive et le bien-être auprès d’un large public en facilitant la découverte de 
    nouvelles activités et en encourageant la motivation entre sportifs.""",
    """Pour créer et partager vos activités sportives et de bien-être, vous devez avoir un compte PRO. Inscrivez-vous sur l’application sportaabe, 
    puis accédez à la section « Créer une activité ». Remplissez les détails de votre activité, ajoutez des photos et des descriptions, puis publiez
    pour que d’autres utilisateurs puissent la rejoindre. Si vous n’avez pas encore de compte PRO, vous pouvez en faire la demande directement via 
    l’application.""",
    """Oui, sportaabe utilise la géolocalisation pour vous aider à trouver des activités sportives et de bien-être près de chez vous. Vous pouvez filtrer
    les résultats par type de sport, niveau de difficulté et d’autres critères pour trouver des activités qui correspondent à vos préférences.""",
    """sportaabe propose une vaste gamme de sports, d’activités de bien-être et de santé, allant des sports d’équipe comme le football et le basketball, 
    aux activités individuelles comme le yoga, le jogging, le fitness, et la méditation. Vous pouvez également trouver des sports de niche et des 
    activités moins courantes, ce qui permet à chacun de trouver quelque chose qui lui plaît.""",
    """L’application sportaabe est disponible sur iOS et Android. Vous pouvez la télécharger gratuitement depuis l’App Store ou Google Play.""",
    """L’inscription à l’application sportaabe est simple et gratuite. Vous pouvez vous inscrire en utilisant votre adresse email ou votre numéro de 
    téléphone. Une fois inscrit, vous aurez accès à toutes les fonctionnalités de l’application pour organiser et rejoindre des activités sportives 
    et de bien-être. Pour créer des activités, vous devez avoir un compte PRO.""",
    """Pour obtenir un compte PRO, inscrivez-vous d’abord sur l’application sportaabe avec un compte standard. Ensuite, accédez à la section des paramètres
    et sélectionnez l’option pour passer à un compte PRO. Suivez les instructions pour compléter votre demande. Une fois approuvé, vous pourrez créer et 
    partager vos propres activités.""",
    """Vous pouvez joindre notre équipe d’assistance via :

        -L’espace « Support » de l’application.

        – Email : contact@sportaabe.com.

        -Téléphone : +237 6 95 27 43 84.
    Nos téléconseillers sont disponibles du lundi au vendredi, de 8h à 17h30.""",
    """Absolument ! Que vous soyez débutant, intermédiaire ou avancé, vous trouverez des activités adaptées à votre niveau..""",
    """:Oui ! sportaabe dispose d’un répertoire de coachs professionnels dans différentes disciplines. Vous pouvez filtrer les coachs selon leur spécialité, localisation et tarifs, puis réserver des séances personnalisées.",
    "Connectez-vous à votre compte.
        - Recherchez une activité ou une infrastructure sportive.
        - Choisissez la date, l’heure et le créneau disponible.
        - Effectuez le paiement via l’application. Une confirmation vous sera envoyée par email et dans votre espace personnel.""",
    """sportaabe accepte les paiements via :

        Mobile Money (MTN)
        Orange Money (Orange).""",
    """Oui ! Si vous êtes propriétaire ou gestionnaire d’un centre sportif, Sportaabe vous permet de :
        - Créer un profil professionnel.
        - Ajouter vos infrastructures et services.
        - Gérer vos réservations directement depuis l’application""",
    """Chaque utilisateur de Sportaabe reçoit un bonus annuel de 10 000 points renouvelés automatiquement chaque 1er janvier. Ces points sont utilisables pour 
    obtenir jusqu’à 15 % de réduction sur le prix des activités choisies, en fonction de son bonus souscrit.""",
    """Pour bénéficier des réductions sur vos activités, vous devez remplir certaines conditions, notamment avoir complété entièrement votre profil sur 
    l’application. Cela garantit que vous pouvez profiter des avantages associés à vos points de manière optimale.""",
    """Vous pouvez obtenir jusqu’à 15 % de réduction sur le prix de l’activité choisie grâce à vos points sportaabe, selon le bonus auquel vous avez souscrit. 
    Assurez-vous de respecter les conditions nécessaires pour activer cet avantage."""
]

# Charger le modèle d'embeddings pour FAISS
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embed_model.encode(documents)
doc_embeddings = np.array(doc_embeddings).astype('float32')

# Créer l'index FAISS
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

# Charger GPT-2 (modèle pré-entraîné et gratuit)
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# ---------------------------
# Fonctions d'assistance

def retrieve_document(query: str, top_k: int = 1):
    query_embedding = embed_model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    # Retourne les documents les plus pertinents
    return [documents[i] for i in indices[0]]

def generate_response(query: str, max_new_tokens: int = 50):
    # Récupérer le document le plus pertinent
    retrieved_text = retrieve_document(query)[0]
    
    # Construire un prompt plus précis
    prompt = f"Voici une question et un contexte associé.\n\nQuestion: {query}\nContexte: {retrieved_text}\nDonne une réponse claire et concise :"

    # Encoder le prompt et générer la réponse avec GPT-2
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)

    # Nettoyer la réponse générée
    response_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    response_text = response_text.split("Réponse:")[-1].strip()  # Ne garder que la réponse utile

    return response_text


# ---------------------------
# Modèle de requête pour FastAPI
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str

# ---------------------------
# Routes de l'API

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        if not request.query:
            raise HTTPException(status_code=400, detail="La requête ne peut pas être vide.")
        answer = generate_response(request.query)
        return ChatResponse(response=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preprocessing chat: {e}")

# Pour lancer le serveur : uvicorn app:app --reload
