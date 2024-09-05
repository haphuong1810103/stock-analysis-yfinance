from mdutils.mdutils import MdUtils
from mdutils import Html
mdText = MdUtils(file_name='text')

mdText.new_header(level=1, title='Overview')
mdText.write("Welcome to <font color='red'>Acme Spinners</font>!\n\n")
mdText.new_paragraph("Visit our website at <a href='#'>acmespinners.ca</a> to view our videos on the latest spinner techniques.\n")
mdText.create_md_file()