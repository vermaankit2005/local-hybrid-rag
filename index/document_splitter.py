from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


class MarkdownDocumentTextSplitterHybrid:

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        # splitter A - cuts at headings
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
        )
        # splitter B - cuts by size
        self.size_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split_documents(self, document: Document) -> list[Document]:
        output = []

        # PASS 1 - loop over each section
        # So as MarkdownHeaderTextSplitter , split the document into sections based on headings, and then for each section
        # and handover the headings in metadata.
        for section in self.header_splitter.split_text(document.page_content):

            # collect whatever headings this section has
            # So section.metadata will look like {'h1': 'Merchandising', 'h2': 'Watches'}

            headers = []
            for key in ("h1", "h2", "h3"):
                if key in section.metadata:
                    headers.append(section.metadata[key])

            breadcrumb = " > ".join(headers)  # "Merchandising > Watches"

            # PASS 2 - cut this section into small pieces
            for piece in self.size_splitter.split_text(section.page_content):
                metadata = document.metadata.copy()
                metadata.update(section.metadata)

                metadata["chunk_id"] = f"{document.metadata['pageid']}_{len(output)}"

                text = f"{breadcrumb}\n\n{piece}" if breadcrumb else piece
                output.append(Document(page_content=text, metadata=metadata))

        return output
