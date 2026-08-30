'use client'

import { useEffect, useRef, useState, type FormEvent } from 'react'
import Card from '@/components/ui/Card'
import Notice from '@/components/ui/Notice'
import Section from '@/components/layout/Section'
import Tag from '@/components/ui/Tag'
import { ApiError, postChat, type ChatEntry } from '@/lib/api'

const SUGGESTED_PROMPTS = [
  'What years do you have baby name data for?',
  'What were the top 10 most popular boy names in 2024?',
  'What are the most popular baby names over the past 10 years?',
  'Is the name David increasing in popularity?',
  'Which baby names have risen in popularity the fastest over the past 10 years?',
  'How many different names were used in 1950 vs 2020?',
]

interface Message extends ChatEntry {
  error?: boolean
}

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === 'user'
  return (
    <div className={`flex gap-2 ${isUser ? 'justify-end' : 'justify-start'}`}>
      {!isUser && (
        <span className="mt-1 shrink-0 font-mono text-[10px] tracking-widest text-[#0ea5e9]">
          AI
        </span>
      )}
      <div
        className={`max-w-[85%] rounded-xl px-4 py-2.5 text-[13px] leading-relaxed ${
          isUser
            ? 'bg-[#0ea5e9]/15 text-[#ededef] shadow-[inset_0_0_0_1px_rgba(14,165,233,0.25)]'
            : message.error
              ? 'bg-red-500/10 text-red-200 shadow-[inset_0_0_0_1px_rgba(239,68,68,0.25)]'
              : 'bg-white/[0.05] text-[#ededef] shadow-[inset_0_0_0_1px_rgba(255,255,255,0.07)]'
        }`}
      >
        <span className="break-words whitespace-pre-wrap">{message.content}</span>
        {message.sql && (
          <details className="mt-2 border-t border-white/[0.08] pt-2">
            <summary className="cursor-pointer font-mono text-[10px] tracking-widest text-[#8a8f98] select-none hover:text-[#0ea5e9]">
              VIEW SQL QUERY
            </summary>
            <pre className="mt-2 overflow-x-auto rounded-lg bg-black/40 p-3 font-mono text-[11px] leading-relaxed whitespace-pre-wrap text-[#7dd3fc]">
              {message.sql}
            </pre>
          </details>
        )}
      </div>
      {isUser && (
        <span className="mt-1 shrink-0 font-mono text-[10px] tracking-widest text-[#8a8f98]">
          YOU
        </span>
      )}
    </div>
  )
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [fatal, setFatal] = useState<string | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }, [messages, busy])

  function send(question: string) {
    const trimmed = question.trim()
    if (!trimmed || busy) return

    const nextMessages: Message[] = [...messages, { role: 'user', content: trimmed }]
    setMessages(nextMessages)
    setInput('')
    setBusy(true)
    setFatal(null)

    postChat(trimmed, messages)
      .then(({ answer, sql }) => {
        setMessages([...nextMessages, { role: 'assistant', content: answer, sql }])
      })
      .catch((e: unknown) => {
        if (e instanceof ApiError && e.status === 503) {
          setFatal(e.message)
          setMessages(messages) // roll back the unanswered question
        } else if (e instanceof ApiError && e.status === 429) {
          setMessages([
            ...nextMessages,
            {
              role: 'assistant',
              content: 'Rate limit reached — please wait a minute before asking again.',
              error: true,
            },
          ])
        } else {
          setMessages([
            ...nextMessages,
            {
              role: 'assistant',
              content: 'Something went wrong answering that. Please try again.',
              error: true,
            },
          ])
        }
      })
      .finally(() => setBusy(false))
  }

  function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault()
    send(input)
  }

  return (
    <Section>
      <div className="mb-8">
        <Tag variant="accent">AI CHAT</Tag>
        <h1 className="mt-3 text-3xl font-semibold tracking-tight text-[#ededef] md:text-4xl">
          Ask the Dataset
        </h1>
        <p className="mt-2 max-w-2xl text-sm leading-relaxed text-[#8a8f98]">
          Ask questions in plain English. The AI translates them into SQL, runs the query against
          the names database, and explains the result — the generated SQL is shown with every
          answer.
        </p>
      </div>

      {fatal && (
        <div className="mb-6">
          <Notice variant="error">{fatal}</Notice>
        </div>
      )}

      <Card variant="glass" className="flex min-h-[60vh] flex-col">
        {/* Messages */}
        <div className="flex-1 space-y-4 overflow-y-auto p-6">
          {messages.length === 0 && (
            <div>
              <p className="mb-3 font-mono text-[11px] tracking-widest text-[#8a8f98]">
                TRY ASKING
              </p>
              <div className="grid gap-2 sm:grid-cols-2">
                {SUGGESTED_PROMPTS.map((prompt) => (
                  <button
                    key={prompt}
                    onClick={() => send(prompt)}
                    className="rounded-lg border border-white/[0.06] bg-white/[0.03] px-4 py-3 text-left text-[13px] text-[#8a8f98] transition-all duration-150 hover:border-[#0ea5e9]/30 hover:bg-[#0ea5e9]/5 hover:text-[#ededef]"
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((message, i) => (
            <MessageBubble key={i} message={message} />
          ))}

          {busy && (
            <div className="flex gap-2">
              <span className="mt-1 shrink-0 font-mono text-[10px] tracking-widest text-[#0ea5e9]">
                AI
              </span>
              <div
                className="rounded-xl bg-white/[0.05] px-4 py-2.5 text-[13px] text-[#8a8f98] shadow-[inset_0_0_0_1px_rgba(255,255,255,0.07)]"
                role="status"
              >
                <span className="animate-[pulse-dot_1.5s_ease-in-out_infinite]">
                  Writing SQL and querying the data…
                </span>
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div className="border-t border-white/[0.06] p-4">
          <form onSubmit={handleSubmit} className="flex items-center gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={busy}
              placeholder="Ask a question about baby names…"
              maxLength={500}
              autoComplete="off"
              spellCheck="false"
              aria-label="Chat message input"
              className="h-10 w-full flex-1 rounded-lg border border-white/[0.08] bg-white/[0.04] px-4 text-[13px] text-[#ededef] placeholder-[#8a8f98]/60 transition-all duration-150 focus:border-[#0ea5e9]/50 focus:bg-white/[0.06] focus:shadow-[0_0_0_3px_rgba(14,165,233,0.15)] focus:outline-none disabled:cursor-not-allowed disabled:opacity-40"
            />
            <button
              type="submit"
              disabled={busy || !input.trim()}
              aria-label="Send message"
              className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-[#0ea5e9] text-[#082f49] shadow-[0_0_0_1px_rgba(14,165,233,0.5),0_2px_8px_rgba(14,165,233,0.3)] transition-all duration-150 hover:bg-[#38bdf8] active:scale-[0.97] disabled:cursor-not-allowed disabled:opacity-30 disabled:hover:bg-[#0ea5e9]"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="h-4 w-4"
                aria-hidden="true"
              >
                <path d="M22 2 11 13" />
                <path d="M22 2 15 22 11 13 2 9l20-7z" />
              </svg>
            </button>
          </form>
        </div>
      </Card>
    </Section>
  )
}
