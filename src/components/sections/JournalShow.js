import { useEffect, useState } from "react"
import { useAuth0 } from "@auth0/auth0-react"
import {
  collection,
  query,
  where,
  orderBy,
  onSnapshot,
} from "@firebase/firestore" // Assuming you're using Firebase Firestore
import { db } from "../../firebase.config"
import AppContext from "../../context/AppContext"
import { useContext } from "react"

function JournalShow() {
  // const [journals, setJournals] = useState([])
  const { journals, setJournals } = useContext(AppContext)

  const { user } = useAuth0()

  useEffect(() => {
    if (user) {
      const journalRef = collection(db, "journals") // Assuming db is your Firestore instance

      const q = query(
        journalRef,
        where("author", "==", user.email),
        orderBy("createdAt", "desc")
      )
      const unsubscribe = onSnapshot(q, (querySnapshot) => {
        const docs = []
        querySnapshot.forEach((doc) => {
          docs.push({ ...doc.data(), id: doc.id })
        })
        console.log(docs)
        setJournals(docs)
      })
      return () => unsubscribe()
    }
  }, [user])

  const journalContent = journals?.map((journal) => {
    const date = journal.createdAt.toDate()
    return (
      <>
        {" "}
        <div
          key={journal.id}
          className="px-3 py-3 bg-gray-100 rounded-md flex flex-col gap-2"
        >
          <p className="text-sm font-bold">{date.toDateString()}</p>
          <p className="text-md"> {journal.text}</p>
        </div>
      </>
    )
  })

  return (
    <div>
      <h1 className="text-2xl font-bold text-center pb-2 pt-3">
        Your Previous Journal Entries
      </h1>
      <div className="flex flex-col gap-4 px-4 my-2">{journalContent}</div>
    </div>
  )
}

export default JournalShow
