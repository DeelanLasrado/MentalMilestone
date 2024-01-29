import { createContext, useState } from "react"

const AppContext = createContext()

function Provider({ children }) {
  const [journals, setJournals] = useState([])
  return (
    <>
      <AppContext.Provider value={{ journals, setJournals }}>
        {children}
      </AppContext.Provider>
    </>
  )
}

export default AppContext
export { Provider }
