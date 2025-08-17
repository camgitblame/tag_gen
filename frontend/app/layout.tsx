import './globals.css'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'Tagline Generator',
  description: 'AI movie tagline generator',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="bg-gray-900">{children}</body>
    </html>
  )
}
