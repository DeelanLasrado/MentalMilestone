import JournalInput from "../components/sections/JournalInput"
import JournalShow from "../components/sections/JournalShow"
import { useAuth0 } from "@auth0/auth0-react"
import AuthenticatedHeader from "../components/sections/AuthenticatedHeader"
import MyCalendar from "../components/sections/calendar"
import Footer from "../components/sections/Footer"

function JournalSection() {
  const { user } = useAuth0()

  return (
    <>
      <div>
        <AuthenticatedHeader />
        <div className="px-2">
          <MyCalendar />
        </div>
        <h1 className="text-4xl pb-2 pt-4 text-center">Journal Section</h1>
        <div className="grid grid-cols-2 max-h-[30rem]">
          <div className="col-start-1 col-span-1">
            <JournalInput />
          </div>
          <div className="col-start-2 col-span-1 max-h-[27.5rem] overflow-x-hidden overflow-scroll">
            <JournalShow />
          </div>
        </div>
        <Footer />
      </div>
    </>
  )
}

export default JournalSection
