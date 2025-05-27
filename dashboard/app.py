import seaborn as sns
from faicons import icon_svg

# Import data from shared.py
from shared import app_dir, df

from shiny import App, reactive, render, ui
from shinyswatch import theme
app_ui=ui.page_fluid(
        ui.tags.head(
            ui.tags.link(
                href="https://cdn.jsdelivr.net/npm/bootswatch@5.3.2/dist/journal/bootstrap.min.css",
                rel="stylesheet"
            )
        ), 
        ui.page_navbar(
            ui.nav_panel("main_page",
                ui.card(
                    ui.card_header("📄 데이터 미리보기 (상위 5개 행)"),
                    ui.output_data_frame("show_head")
                )
            ),
        title=ui.tags.a("타이틀", href="/", style="text-decoration:none; color:inherit;"),
        theme = theme.journal
                
        )
    )


def server(input, output, session):
    @output
    @render.data_frame
    def show_head():
        return df.head()



app = App(app_ui, server)
