import { useRef } from "react"
import {
  addDoc,
  collection,
  query,
  where,
  orderBy,
  limit,
  getDocs,
} from "firebase/firestore"
import { db } from "../../firebase.config"
import { useAuth0 } from "@auth0/auth0-react"
import axios from "axios"

function JournalInput() {
  const { user } = useAuth0()
  const textAreaRef = useRef(null)

  const handleEmailSend = async () => {
    const journalRef = collection(db, "journals") // Assuming db is your Firestore instance

    const q = query(
      journalRef,
      where("author", "==", user.email),
      orderBy("createdAt", "desc"),
      limit(3)
    )

    const querySnapshot = await getDocs(q)

    const docs = []
    querySnapshot.forEach((doc) => {
      docs.push({ ...doc.data(), id: doc.id })
    })

    console.log(docs)

    let cnt = 0
    docs.forEach((doc) => {
      if (doc.sentiment === "negative") {
        cnt++
      }
    })

    if (cnt === 3)
      axios.post("http://127.0.0.1:5000/alert", {
        email: user.email,
        name: user.name,
        data: docs,
      })
  }

  const handleSubmit = async (e) => {
    e.preventDefault();

    const journalRef = collection(db, "journals");

    const response = await axios.get("http://127.0.0.1:5000/message", {
        params: {
            message: textAreaRef.current.value,
        },
    });

    //const { text } = response.data;
    //console.log(text)
    const text = response.data.sentiment;
    console.log("sentiments "+text)
    

    // Check if sentiment is undefined
    if (typeof text === 'undefined') {
        console.error('Sentiment analysis result is undefined.');
        // Handle the error, set a default sentiment, or skip document creation
        return;
    }

    addDoc(journalRef, {
        text: textAreaRef.current.value,
        author: user.email,
        sentiment: text,
        createdAt: new Date(),
    });

    handleEmailSend();
    textAreaRef.current.value = "";
};



  return (
    <>
      <form onSubmit={handleSubmit}>
        <div className="flex flex-col gap-2 items-start p-2">
          <label
            htmlFor="name"
            className="text-2xl font-bold text-center w-full"
          >
            How was your day? 😁
          </label>
          <textarea
            name="name"
            id="name"
            rows="14"
            className="w-full p-3 rounded-md outline"
            ref={textAreaRef}
            placeholder="You can share your thoughts with me."
          ></textarea>
          <button className="px-3 py-2 bg-green-600 text-white rounded-md shadow mx-auto">
            Submit
          </button>
        </div>
      </form>
    </>
  )
}

export default JournalInput
