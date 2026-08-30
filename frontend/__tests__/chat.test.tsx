import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MessageBubble, type Message } from '@/app/chat/page'

describe('MessageBubble', () => {
  it('renders user messages with YOU label and plain text', () => {
    const message: Message = {
      role: 'user',
      content: 'What were the top 5 names in 2024?',
    }
    render(<MessageBubble message={message} />)
    expect(screen.getByText('What were the top 5 names in 2024?')).toBeInTheDocument()
    expect(screen.getByText('YOU')).toBeInTheDocument()
    expect(screen.queryByText('AI')).not.toBeInTheDocument()
  })

  it('renders assistant messages with AI label and markdown formatting', () => {
    const markdown =
      'The top name was **Liam** with *over 20,000* births. ***Remarkable growth*** since 2010.'
    const message: Message = {
      role: 'assistant',
      content: markdown,
    }
    const { container } = render(<MessageBubble message={message} />)

    expect(screen.getByText('AI')).toBeInTheDocument()
    expect(screen.queryByText('YOU')).not.toBeInTheDocument()

    const strong = container.querySelectorAll('strong')
    expect(strong.length).toBeGreaterThanOrEqual(2)
    expect(screen.getByText('Liam')).toBeInTheDocument()
    expect(screen.getByText('Remarkable growth')).toBeInTheDocument()

    const em = container.querySelectorAll('em')
    expect(em.length).toBeGreaterThanOrEqual(2)
    expect(screen.getByText('over 20,000')).toBeInTheDocument()
  })

  it('renders lists, blockquotes, inline code, and links in assistant responses', () => {
    const markdown = `> SSA Historical dataset analysis.

Top 3 boy names:
1. **Liam** - 20,802
2. **Noah** - 18,995
3. **Oliver** - 14,741

Check \`total_count\` or visit [SSA Website](https://www.ssa.gov).`

    const message: Message = {
      role: 'assistant',
      content: markdown,
    }
    const { container } = render(<MessageBubble message={message} />)

    expect(container.querySelector('blockquote')?.textContent).toContain('SSA Historical dataset')
    expect(container.querySelector('ol')?.querySelectorAll('li').length).toBe(3)
    expect(container.querySelector('code')?.textContent).toBe('total_count')

    const link = container.querySelector('a')
    expect(link?.getAttribute('href')).toBe('https://www.ssa.gov')
    expect(link?.getAttribute('target')).toBe('_blank')
    expect(link?.getAttribute('rel')).toBe('noopener noreferrer')
  })

  it('safely escapes raw HTML and prevents script/image tag injection', () => {
    const malicious = 'Top name <script>alert("xss")</script><img src=x onerror=alert(1)> text'
    const message: Message = {
      role: 'assistant',
      content: malicious,
    }
    const { container } = render(<MessageBubble message={message} />)
    expect(container.querySelector('script')).toBeNull()
    expect(container.querySelector('img')).toBeNull()
  })

  it('renders SQL query in collapsible details when present', () => {
    const message: Message = {
      role: 'assistant',
      content: 'Here is the data for 2024.',
      sql: 'SELECT name, total_count FROM names WHERE year = 2024 LIMIT 10;',
    }
    render(<MessageBubble message={message} />)

    expect(screen.getByText('VIEW SQL QUERY')).toBeInTheDocument()
    expect(
      screen.getByText('SELECT name, total_count FROM names WHERE year = 2024 LIMIT 10;')
    ).toBeInTheDocument()
  })

  it('renders error messages with error styling as plain text', () => {
    const message: Message = {
      role: 'assistant',
      content: 'Rate limit reached — please wait a minute.',
      error: true,
    }
    render(<MessageBubble message={message} />)
    expect(screen.getByText('Rate limit reached — please wait a minute.')).toBeInTheDocument()
  })
})
