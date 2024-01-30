import { initializeApp } from "firebase/app"
import { getFirestore } from "firebase/firestore"

const firebaseConfig = {
  apiKey: "AIzaSyC8jqO7C94GcEmN4C9xiikW4K7FmRIuYIg",
  authDomain: "localhost",
  projectId: "mentalmilestone-a52f3",
  storageBucket: "gs://mentalmilestone-a52f3.appspot.com",
  messagingSenderId: "79553728983",
  appId: "1:79553728983:web:eff4c14dd8bbb6dd60fb92",
  measurementId: "G-9FPCMETH0B",
}

export const app = initializeApp(firebaseConfig)
export const db = getFirestore(app)
