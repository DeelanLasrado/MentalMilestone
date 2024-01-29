import { Calendar, momentLocalizer } from "react-big-calendar"
import moment from "moment"
import AppContext from "../../context/AppContext"
import { useContext } from "react"
import "react-big-calendar/lib/css/react-big-calendar.css"
const localizer = momentLocalizer(moment)

function MyCalendar() {
  const { journals } = useContext(AppContext)

  console.log(journals)
  const myEventsList = journals?.map((journal) => ({
    title: journal.sentiment,
    start: journal.createdAt.toDate(),
    end: journal.createdAt.toDate(),
    sentiment: journal.sentiment,
  }))

  const dayPropGetter = (date) => {
    const event = myEventsList.find((e) => moment(e.start).isSame(date, "day"))
    console.log(event)
    if (event && event.sentiment === "positive") {
      return { style: { backgroundColor: "green" } }
    } else if (
      event &&
      (event.sentiment === "negative" || event.sentiment === "Negative.")
    ) {
      return { style: { backgroundColor: "red" } }
    } else if (event && event.sentiment === "neutral") {
      return { style: { backgroundColor: "yellow" } }
    } else {
      return { style: { backgroundColor: "white" } }
    }
  }

  return (
    <div className="h-[25rem]">
      <Calendar
        localizer={localizer}
        events={myEventsList}
        startAccessor="start"
        endAccessor="end"
        dayPropGetter={dayPropGetter}
      />
    </div>
  )
}

export default MyCalendar
